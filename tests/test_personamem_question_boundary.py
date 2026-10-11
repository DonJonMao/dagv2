"""PersonaMem question boundaries, using synthetic text and loopback responses."""
import sys
from types import SimpleNamespace

import pytest

from dagbt import personamem, runner


def _full_question(question, marker):
    return personamem.format_question(question, [f"{marker} {letter}" for letter in "ABCD"])


def _reader(monkeypatch):
    def messages(**kwargs):
        return [{"role": "system", "content": "You are a long-document QA reader."},
                {"role": "user", "content": "Question:\n" + kwargs["question"] + "\n\n"
                 "Answer with the shortest exact phrase supported by the context passages."}]
    reader = SimpleNamespace(reader_messages=messages, parse_answer=lambda text: text)
    monkeypatch.setitem(sys.modules, "reader", reader)
    runner.configure_personamem_reader()
    return reader


def _experiment(reader, solve):
    class PlanError(ValueError):
        pass

    observed = []

    def plan(question, _calls):
        observed.append(("planner", question))
        # The per-call override must start at solve, not at planner/archive.
        assert "Question:\nProbe\n\n" in reader.reader_messages(question="Probe")[-1]["content"]
        return {"steps": []}

    def archive(question, *_):
        observed.append(("archive", question))
        return ["memory"], []

    def failure(unit, status, error):
        return {"unit_id": unit, "answer": {"status": status, "error": str(error)}}

    return SimpleNamespace(make_plan=plan, archive=archive, native=SimpleNamespace(solve=solve),
                           PlanError=PlanError, failure=failure), observed


def _result(unit):
    return {"unit_id": unit, "ranking": {}, "answer": {"status": "ok", "prediction": "(a)"}}


@pytest.mark.parametrize("first_failure", [None, "uncaught", "context_budget"])
def test_original_reader_override_is_local_to_solve_and_resets_between_questions(
        tmp_path, monkeypatch, first_failure):
    reader = _reader(monkeypatch)
    rendered = []
    requests = ["Which drink should I buy?", "Which walk should I take?"]
    full = [_full_question(requests[0], "FIRST_OPTIONS"), _full_question(requests[1], "SECOND_OPTIONS")]

    def solve(row, *_):
        index = len(rendered)
        assert row["question"] == requests[index]
        rendered.append(reader.reader_messages(question=row["question"])[-1]["content"])
        assert full[index] in rendered[-1]
        assert full[1 - index] not in rendered[-1]
        if index == 0 and first_failure == "uncaught":
            raise RuntimeError("synthetic solve failure")
        if index == 0 and first_failure == "context_budget":
            raise AssertionError(("context_budget", 20000))
        return _result(row["unit_id"])

    experiment, observed = _experiment(reader, solve)
    resources = ({}, ["memory"], None, None, None)
    for index, question in enumerate(requests):
        calls = SimpleNamespace(output=tmp_path / str(index))
        invoke = lambda: runner.original_question(
            {"id": str(index), "question": question}, resources, calls, experiment,
            reader_question=full[index])
        if index == 0 and first_failure == "uncaught":
            with pytest.raises(RuntimeError, match="synthetic solve failure"):
                invoke()
        else:
            result = invoke()
            expected = "context_overflow" if index == 0 and first_failure == "context_budget" else "ok"
            assert result["answer"]["status"] == expected
        probe = reader.reader_messages(question="After solve")[-1]["content"]
        assert "Question:\nAfter solve\n\n" in probe
        assert not any(value in probe for value in full)
    assert observed == [(stage, question) for question in requests for stage in ("planner", "archive")]
    assert len(rendered) == 2


def test_original_without_reader_override_preserves_question(tmp_path, monkeypatch):
    reader = _reader(monkeypatch)
    question = "Which drink should I buy?"

    def solve(row, *_):
        assert row["question"] == question
        assert "Question:\n" + question + "\n\n" in reader.reader_messages(question=question)[-1]["content"]
        return _result(row["unit_id"])

    experiment, observed = _experiment(reader, solve)
    result = runner.original_question({"id": "plain", "question": question},
                                      ({}, ["memory"], None, None, None),
                                      SimpleNamespace(output=tmp_path), experiment)
    assert result["answer"]["status"] == "ok"
    assert observed == [("planner", question), ("archive", question)]


def test_loopback_reproduces_option_response_for_old_mcq_planner_input(tmp_path, monkeypatch):
    from dagbt.budget import Ledger
    from dagbt.engine import legacy_modules
    from dagbt.transport import Transport
    from test_personamem_runner import _rank_future_first_models

    for name in ("DAG_LLM_API_KEY", "DAG_EMBED_API_KEY", "DAG_RERANK_API_KEY", "BRIDGETREE_CHAT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    experiment, _, _ = legacy_modules()
    credentials = tmp_path / "empty_credentials.json"
    runner.save(credentials, {})
    question = "Which drink should I buy?"
    with _rank_future_first_models([], [], obey_leaked_answer_instruction=True) as (endpoint, requests, errors):
        config = runner.load(runner.ROOT / "configs/paired.example.json")
        config.update(llm_base_url=endpoint + "/v1", credentials_file=str(credentials),
                      request_timeout_seconds=10, max_identical_attempts=1)
        monkeypatch.setattr(experiment, "CONFIG", config)
        calls = Transport("synthetic_question", tmp_path / "requests", config, Ledger({}), None)
        with pytest.raises(experiment.PlanError, match="^planner_json$"):
            experiment.make_plan(_full_question(question, "CANDIDATE"), calls)
        plan = experiment.make_plan(question, calls)
        assert plan["steps"][0]["question"] == question
        assert not errors, errors
        assert len(requests) == 2
        records = [runner.load(path) for path in (tmp_path / "requests/requests").glob("*.json")]
        bad = next(record for record in records
                   if personamem.ANSWER_INSTRUCTION in record["payload"]["messages"][1]["content"])
        assert bad["response"]["choices"][0] == {"message": {"content": "(a)"}, "finish_reason": "stop"}
