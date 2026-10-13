"""Real-spawn regressions for stop arriving before Process.start registers PID."""
import multiprocessing as mp
import multiprocessing.util
import os
import signal
import time

import pytest

from dagbt.runner import Worker


def waiting_worker(connection, dataset, arm, config):
    blocked = (signal.pthread_sigmask(signal.SIG_BLOCK, [])
               if hasattr(signal, 'pthread_sigmask') else set())
    connection.send({'type': 'ready', 'hashes': {'blocked_sigterm': signal.SIGTERM in blocked}})
    while True:
        try:
            if connection.recv() is None:
                return
        except EOFError:
            return


def stop_handler(signum, frame):
    raise KeyboardInterrupt('synthetic user stop')


def raw_cleanup(pid):
    """Keep the negative regression safe even before the production fix."""
    if pid is None:
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


@pytest.mark.skipif(os.name != 'posix', reason='POSIX spawn boundary regression')
@pytest.mark.parametrize('stop_signal', [signal.SIGTERM, signal.SIGINT])
def test_signal_between_os_spawn_and_process_registration_is_replayed_after_registration(monkeypatch, stop_signal):
    worker = Worker('synthetic', 'original', {'experiment': {'worker_startup_timeout_seconds': 5}},
                    target=waiting_worker)
    original = multiprocessing.util.spawnv_passfds
    spawned = []

    def spawn_then_stop(path, args, passfds):
        pid = original(path, args, passfds)
        if any('spawn_main' in str(arg) for arg in args):
            spawned.append(pid)
            # Popen._launch has an OS child, but Process._popen and .pid have
            # not yet been assigned. Raising here used to orphan the child
            # and Worker.close replaced KeyboardInterrupt with AssertionError.
            os.kill(os.getpid(), stop_signal)
        return pid

    previous = signal.signal(stop_signal, stop_handler)
    monkeypatch.setattr(multiprocessing.util, 'spawnv_passfds', spawn_then_stop)
    try:
        with pytest.raises(KeyboardInterrupt, match='synthetic user stop'):
            worker.run({'id': 'synthetic'}, 'unused-output')
        assert spawned
        assert worker.process is None and worker.connection is None
        assert signal.getsignal(stop_signal) is stop_handler
        # start's error path must terminate AND reap its real spawned child.
        with pytest.raises(ChildProcessError):
            os.waitpid(spawned[0], os.WNOHANG)
    finally:
        signal.signal(stop_signal, previous)
        for pid in spawned:
            raw_cleanup(pid)
        if worker.connection is not None:
            worker.connection.close()


def test_close_accepts_process_constructed_but_not_started():
    worker = Worker('synthetic', 'original', {}, target=waiting_worker)
    worker.process = mp.get_context('spawn').Process(target=waiting_worker)
    parent, child = mp.get_context('spawn').Pipe()
    worker.connection = parent
    try:
        worker.close()
        worker.close()
        assert parent.closed
        assert worker.process is None and worker.connection is None
    finally:
        parent.close()
        child.close()


def test_spawned_worker_does_not_inherit_blocked_stop_signal():
    worker = Worker('synthetic', 'original', {'experiment': {'worker_startup_timeout_seconds': 5}},
                    target=waiting_worker)
    previous = signal.signal(signal.SIGTERM, stop_handler)
    try:
        worker.start()
        process = worker.process
        assert worker.hashes['blocked_sigterm'] is False
        assert signal.getsignal(signal.SIGTERM) is stop_handler
        start = time.monotonic()
        worker.close()
        assert time.monotonic() - start < 2
        assert process.exitcode is not None
    finally:
        signal.signal(signal.SIGTERM, previous)
        worker.close()


def test_spawn_pickle_failure_preserves_exception_and_restores_stop_handler():
    # Deliberately unpicklable local target fails before an OS worker exists.
    worker = Worker('synthetic', 'original', {}, target=lambda *args: None)
    previous = signal.signal(signal.SIGTERM, stop_handler)
    try:
        with pytest.raises((AttributeError, TypeError), match='pickle'):
            worker.start()
        assert signal.getsignal(signal.SIGTERM) is stop_handler
        assert worker.process is None and worker.connection is None
        worker.close()
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_stop_while_waiting_for_ready_also_closes_started_worker(monkeypatch):
    worker = Worker('synthetic', 'original', {}, target=waiting_worker)
    spawned = []
    def interrupt_ready(timeout):
        spawned.append(worker.process.pid)
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError('Stop handler should have interrupted readiness wait')
    monkeypatch.setattr(worker, '_receive', interrupt_ready)
    previous = signal.signal(signal.SIGTERM, stop_handler)
    try:
        with pytest.raises(KeyboardInterrupt, match='synthetic user stop'):
            worker.start()
        assert signal.getsignal(signal.SIGTERM) is stop_handler
        assert worker.process is None and worker.connection is None
        with pytest.raises(ChildProcessError):
            os.waitpid(spawned[0], os.WNOHANG)
    finally:
        signal.signal(signal.SIGTERM, previous)
        for pid in spawned:
            raw_cleanup(pid)
        worker.close()
