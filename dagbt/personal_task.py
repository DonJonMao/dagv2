"""Public PersonaMem task semantics, independent of evaluation labels.

Facts and preferences are intermediate evidence. The terminal is a semantic
choice of a response to the original utterance, never a fact certificate.
"""
from .reasoning import ProtocolError


PERSONAL_PLAN_GUIDANCE = '''
PERSONAMEM RESPONSE-SELECTION CONTRACT (takes precedence over generic QA examples):
The user utterance is context for selecting a personalized conversational response.
Even a statement such as "I made a table" needs an appropriate response, not merely
the name of an object or an event. Public candidate responses are supplied only at
the terminal and its audit; do not invent candidates while planning.
Plan 1..5 useful history tasks followed by ONE semantic response-selection terminal.
The terminal question MUST be the exact original user utterance, unchanged. Its
execution is compose, with the needed history tasks as inputs. Its ONLY variable
has type=text; program.expression MUST reference that terminal variable. The
output_contract MUST be mode=semantic, result_type=text, explanation=semantic.
The terminal represents a response now; give it a response scope distinct from
the scopes of the historical facts. No symbolic certificate establishes which
personalized response is best. Do not use rule_binding or a deterministic
task_expression to replace this response selection.
Every nonterminal variable MUST have an explicit scope with entity_scope and
time_scope; its history/evidence scope must differ from the terminal response scope.
All retrieval tasks obtain evidence in this user's conversation before the cutoff.
Never require retrieval of external new-book catalogues, current event listings,
new workshop availability, or a previously executed future recommendation. Retrieve
interests, experiences, constraints and changes that help judge such suggestions.
An unrequested detail (e.g. dish name, festival name, exact packing list) must not
become a mandatory prerequisite just because it is absent from the utterance.
Intermediate compose tasks summarize sourced history; recommendations are judged
at the terminal. Do not impose a universal exhaustive preference profile. Define
small query-relevant evidence needs and bounded values rather than an entire
history as one supposedly complete set. Multiple sources can jointly support a
value; incomplete windows are partial evidence, not mutually exclusive whole sets.
Keep past and later attitudes, their conditions and supporting quotations distinct.
An earlier dislike and later successful use may both be true; do not label the
earlier attitude as current or apply unconditional latest-wins. The terminal must
compare personal fit, constraints and timeline, not just keyword or event overlap.
Every planned task must contribute to the designated terminal. Omit redundant tasks.
The current user utterance is given input. Do not require the memory to re-prove
an event or attribute explicitly stated there. Seek the relevant prior preferences,
experiences and changes that contextualize it instead.
History tasks need sufficient query-relevant evidence, not an exhaustive inventory
of every past event, preference or experience. For open-ended background, ask for
a concise sourced text summary of relevant examples and changes. Use a set variable
only when its task explicitly defines a bounded whole set that can be established
from the available history; a single witnessed member does not establish that whole.
For an utterance such as "I tried X", do not make listing everything the user has
ever tried a prerequisite. Retrieve the prior attitudes or experiences needed to
understand X in context. A separate history-summary node is optional, not mandatory.
'''

PERSONAL_HISTORY_GUIDANCE = '''
CURRENT PHASE: HISTORY EVIDENCE. Complete only the supplied historical task and
requested variable scopes. This is not the final response-selection stage.
For retrieval, extract concrete statements supported by the supplied original
history. For an intermediate compose task, summarize those historical facts,
preferences, constraints and changes using the supplied parent versions and
original quotations. A history summary describes what the sources establish;
it must not address the user, offer a new recommendation or draft a reply.
The original user utterance provides query context. It is not an instruction to
answer it at this stage or to retrieve an exact duplicate of what it already says.
Public response options are not available here. Do not invent, compare or refer
to options A/B/C/D. Return exactly the slots object; no final_prediction or
semantic_answer fields belong in a history-evidence response.
Keep each value within its declared type, entity and time scope. Preserve earlier
and later attitudes, conditions and supporting quotations; a historical dislike
must not silently replace a later documented change. Assistant advice alone does
not establish an adopted user preference. Do not invent external catalogues,
availability or past execution of a future suggestion. A partial observation must
cite its actual clue. Use unresolved only for a specific condition still necessary
to establish this requested historical value, not hypothetical exhaustive coverage.
For one text-summary variable, synthesize compatible query-relevant facts into one
sufficient source-grounded whole summary; do not report each event as a different
complete value of that variable. Multiple supports must be alternative sufficient
proofs of the same whole value. Keep partial evidence, negative experiences and
valid counters explicit when a complete scoped summary is not established.
Positive and negative experiences can both support a historical summary. A sourced
dislike is useful history, not automatically a contradiction observation: that
stance is reserved for challenging an identified existing fact route. Preserve
the difference between past enjoyment, later frustration and their conditions in
the value and citations; do not summarize them as an unconditional current interest.
Re-read any supplied evidence obligations and their original sources explicitly;
do not drop a source or contradiction merely because a prior value is available.
'''

PERSONAL_READING_GUIDANCE = '''
CURRENT PHASE: FINAL RESPONSE SELECTION.
PersonaMem asks which response best addresses the original user utterance using
personal history. A source-supported fact alone is not a completed answer.
For nonterminal tasks keep query-relevant historical observations, stated reasons,
constraints and changes in preference with their own source/time scope. Assistant
advice is not an adopted user preference unless user evidence establishes adoption.
At response selection compare ALL public_options for personalization, relevance,
contradictions and current applicability. Candidate options are not evidence of
past events or user preferences. A prospective suggestion need not have occurred
in the memory: its personal fit must be supported, without asserting unavailable
external facts such as live availability. Distinguish suggestion from history.
For a supported selection, the sole terminal variable's value and semantic_answer
MUST both be the EXACT text of the selected public_options item, without its label.
final_prediction alone is its letter. Cite the history that justifies the choice;
do not substitute a fact summary, paraphrase or generic praise for the response.
If the evidence cannot discriminate the options, leave the answer unresolved.
'''

PERSONAL_AUDIT_GUIDANCE = '''
CURRENT PHASE: RESPONSE AUDIT.
Audit the proposed response selection against the ORIGINAL utterance, ALL public
options and supplied original history, including relevant counterevidence. Check
whether another option better satisfies the user's interests, experiences,
constraints and preference timeline. Confirming a narrow fact (object, event or
name) does not validate a generic reply. Do not treat an option as memory evidence.
The selected response is a semantic assessment, not a formal proof of correctness.
Report conflict/unknown with sourced corrections and unresolved issues when its
personal fit is unsupported; preserve older and newer evidence with their scopes.
Do not silently change a label in the audit: correction requires terminal
re-evaluation and another audit of the revised selection.
'''


def validate_personal_plan(plan, question):
    """Reject structurally wrong objectives before the first retrieval call."""
    contract, program = plan['output_contract'], plan['program']
    if any(contract[key] != expected for key, expected in (
            ('mode', 'semantic'), ('result_type', 'text'), ('explanation', 'semantic'))):
        raise ProtocolError('PersonaMem requires semantic/text response selection, not symbolic fact certification')
    tasks = {step['output_slot']: step for step in plan['steps']}
    final = plan['final_node_id']
    terminal = tasks[final]
    if terminal['execution'] != 'compose' or not terminal['inputs']:
        raise ProtocolError('PersonaMem terminal must compose personal-history evidence into a response selection')
    if terminal['question'] != question:
        raise ProtocolError('PersonaMem terminal question must equal the exact original user utterance')
    variables = program['variables']
    terminal_variables = [v for v, declaration in variables.items() if declaration['demand'] == final]
    if (len(terminal_variables) != 1
            or variables[terminal_variables[0]]['type'] != 'text'
            or program['expression'] != {'op': 'var', 'id': terminal_variables[0]}):
        raise ProtocolError('PersonaMem output must be the sole text variable of the response-selection terminal')
    if program.get('rule_binding') or terminal_variables[0] in program.get('task_expressions', {}):
        raise ProtocolError('PersonaMem response selection cannot be replaced by a sourced rule or deterministic fact expression')
    closure = {final}
    pending = [final]
    while pending:
        for parent in tasks[pending.pop()]['inputs']:
            if parent not in closure:
                closure.add(parent)
                pending.append(parent)
    if closure != set(tasks):
        raise ProtocolError('Every PersonaMem history task must contribute to the response-selection terminal')
    demands = {declaration['demand'] for declaration in variables.values()}
    if demands != set(tasks):
        raise ProtocolError('Every PersonaMem task needs an explicit typed evidence or response variable')
    response_scope = {key: contract[key] for key in ('entity_scope', 'time_scope')}
    for variable, declaration in variables.items():
        if declaration['demand'] != final and (
                'scope' not in declaration or declaration['scope'] == response_scope):
            raise ProtocolError('PersonaMem history variable ' + variable
                                + ' needs an explicit evidence scope distinct from the response scope')
    return plan
