#!/usr/bin/env python3
"""Task classes for the trace collection, shared by every harness.

§12.8 is the limitation that threatens the premise rather than the conclusion:
every motivating number in the paper -- 3.9% hit rate, 77% false completions,
duty cycle 0.277, the 68:1 input/output ratio, zero cross-member prefix -- comes
from two runs of one task class on one model. We have already been burnt once by
exactly this: the ring was reported as the measured shape of peer collaboration
until a wider sample showed the speaking order is a property of the harness, and
that correction changed the abstract and the last piece of the mechanism.

So the task bank exists to vary the things that could plausibly move those
numbers, and each class is chosen for a specific reason rather than for variety:

`design`   the class already collected, kept byte-identical so the new runs are
           comparable to the ones in §3. Every member is told to restate and
           extend, which maximises private history and minimises deferral.

`debug`    the adversarial case for *duty cycle and false completions*. Members
           form hypotheses and wait for each other's evidence, so turns are
           short, deferral is natural, and if 77% false completions is an
           artefact of a task that never lets an agent stop, this is where it
           shows.

`review`   the adversarial case for *cross-member prefix sharing*, the premise
           behind "sharing is not the lever, residency is". Here every member
           reads the same artefact, which is the most favourable setting we can
           construct for shared prefixes. If cross-member sharing is ever
           non-trivial it will be here, and measuring near zero again is a much
           stronger statement than measuring it on `design` alone.

Roles carry distinct system prompts throughout, because identical preambles
would manufacture the very cross-member reuse we are trying to measure.
"""

DESIGN_ROLES = [
    ("architect",
     "You are the systems architect. You own the overall design. Every turn, "
     "restate the current design in full, then extend it with one new "
     "subsystem in concrete detail. Never summarise away earlier detail."),
    ("performance",
     "You are the performance engineer. Every turn, take the architect's most "
     "recent design and work out its throughput, latency and memory "
     "behaviour quantitatively, carrying forward all earlier analysis."),
    ("reliability",
     "You are the reliability engineer. Every turn, enumerate the failure "
     "modes of everything proposed so far, including partial failures, and "
     "keep the full running list."),
    ("security",
     "You are the security engineer. Every turn, threat-model everything "
     "proposed so far and keep the full running threat list."),
    ("operator",
     "You are the operator. Every turn, describe how to deploy, monitor and "
     "roll back everything proposed so far, keeping the full runbook."),
    ("data",
     "You are the data engineer. Every turn, specify the schemas, retention "
     "and migration path for everything proposed so far, in full."),
    ("client",
     "You are the client engineer. Every turn, specify the client-side API "
     "surface and compatibility story for everything proposed so far."),
    ("capacity",
     "You are the capacity engineer. Every turn, size the fleet, the KV pool "
     "and the admission budget for everything proposed so far, in full."),
]

DESIGN_TASK = (
    "Design a multi-tenant GPU inference platform that serves teams of "
    "collaborating LLM agents. Work through it together, each from your own "
    "role. Be concrete and quantitative. Do not stop early and do not defer "
    "to each other -- every one of you must contribute substantive new "
    "material on every turn."
)

DEBUG_ROLES = [
    ("reporter",
     "You are the on-call engineer who caught the incident. You hold the "
     "symptoms and the timeline. Answer questions from the others with "
     "specific observations, and say plainly when you do not have the data "
     "someone asks for."),
    ("hypothesis",
     "You form falsifiable hypotheses about the root cause. Propose one at a "
     "time, state exactly what observation would rule it out, and wait for "
     "that observation before moving on."),
    ("instrumentation",
     "You decide what to measure. Given a hypothesis, specify the precise "
     "instrumentation that would falsify it, and report what it would cost to "
     "collect."),
    ("codereader",
     "You reason about the implementation. When a hypothesis implicates a "
     "component, walk the relevant code path and say whether the mechanism is "
     "even possible."),
    ("verifier",
     "You keep the eliminated-candidate list and refuse conclusions that the "
     "evidence does not support. Challenge any claim that has not been "
     "falsified or confirmed by a stated observation."),
    ("historian",
     "You track what changed recently and correlate it with the timeline."),
    ("scribe",
     "You maintain the running investigation log: hypotheses, evidence, and "
     "what remains open."),
]

DEBUG_TASK = (
    "A GPU inference service has started returning p99 latencies 40x its p50 "
    "after a deploy, but only between 02:00 and 04:00 UTC, and only on hosts "
    "serving more than eight tenants. Throughput and error rates are normal. "
    "Find the root cause. Work as an investigation: form hypotheses, ask each "
    "other for the specific evidence that would rule them out, and do not "
    "guess when you can ask."
)

REVIEW_ROLES = [
    ("author",
     "You wrote the document under review. Defend or revise specific claims "
     "when challenged, and quote the passage you are changing."),
    ("factchecker",
     "You verify every quantitative claim in the document against the "
     "evidence it cites, and flag each one that is unsupported."),
    ("structure",
     "You judge whether the argument's structure supports its conclusion, "
     "quoting the passages that do the load-bearing work."),
    ("adversary",
     "You are the hostile reviewer. Construct the strongest case that the "
     "document's central claim is wrong, quoting the text you attack."),
    ("methods",
     "You audit the methodology: what was measured, what was assumed, and "
     "which stated conclusions the measurements cannot support."),
    ("related",
     "You place the work against prior art and identify claims of novelty "
     "that prior work already covers."),
    ("editor",
     "You maintain the running list of required revisions with the exact "
     "passage each one applies to."),
]

REVIEW_TASK = (
    "Review this claim and the argument for it: \"LRU is the wrong eviction "
    "policy for LLM agent teams, because team members take turns and the "
    "member that has been silent longest is the one about to speak, so LRU "
    "evicts exactly the session that is about to be needed. Therefore serving "
    "systems should partition cache capacity per team.\" Each of you review it "
    "from your own angle, quote the specific part you are addressing, and keep "
    "your full running list of findings in every turn."
)

TASKS = {
    "design": (DESIGN_ROLES, DESIGN_TASK),
    "debug": (DEBUG_ROLES, DEBUG_TASK),
    "review": (REVIEW_ROLES, REVIEW_TASK),
}


def get(name, members):
    if name not in TASKS:
        raise SystemExit(f"unknown task {name!r}; have {sorted(TASKS)}")
    roles, task = TASKS[name]
    if members > len(roles):
        raise SystemExit(f"task {name!r} defines only {len(roles)} roles")
    return roles[:members], task


# CrewAI wants (role, goal, backstory) rather than (name, system prompt), so it
# gets its own view of the same three classes. `design` is byte-identical to
# what `harness_crewai.py` defined inline, which is what keeps the collected
# k=5 ring trace comparable to everything collected after it.
CREW = {
    "design": ([
        ("systems architect", "own the overall design of the platform",
         "You restate the current design in full each turn, then extend it with "
         "one new subsystem in concrete detail. You never summarise away detail."),
        ("performance engineer", "quantify throughput, latency and memory",
         "You carry forward all earlier analysis and add new quantitative work."),
        ("reliability engineer", "enumerate and mitigate failure modes",
         "You keep the full running list of failure modes, including partial ones."),
        ("security engineer", "threat-model the whole system",
         "You keep the full running threat list and never drop earlier entries."),
        ("operator", "deploy, monitor and roll back the system",
         "You keep the full runbook and extend it every turn."),
        ("data engineer", "own schemas, retention and migration",
         "You keep the full data design and extend it every turn."),
        ("client engineer", "own the client API surface and compatibility",
         "You keep the full client contract and extend it every turn."),
    ],
        "Design a multi-tenant GPU inference platform that serves teams of "
        "collaborating LLM agents."),

    "debug": ([
        ("on-call engineer", "hold the symptoms and the timeline",
         "You answer with specific observations and say plainly when you do "
         "not have the data someone asks for."),
        ("hypothesis lead", "form falsifiable hypotheses about the cause",
         "You propose one hypothesis at a time and state exactly what "
         "observation would rule it out."),
        ("instrumentation engineer", "decide what to measure",
         "Given a hypothesis you specify the instrumentation that would "
         "falsify it and what collecting it would cost."),
        ("implementation reader", "judge whether a mechanism is possible",
         "You walk the relevant code path and say whether the proposed "
         "mechanism could produce the observed symptom at all."),
        ("verifier", "refuse conclusions the evidence does not support",
         "You keep the eliminated-candidate list and challenge any claim not "
         "backed by a stated observation."),
        ("change historian", "correlate recent changes with the timeline",
         "You track what shipped and when, and line it up against symptoms."),
        ("scribe", "maintain the investigation log",
         "You keep hypotheses, evidence and open questions in one place."),
    ],
        "Find the root cause of a GPU inference service whose p99 latency went "
        "to 40x its p50 after a deploy, but only between 02:00 and 04:00 UTC "
        "and only on hosts serving more than eight tenants, with throughput "
        "and error rates unchanged."),

    "review": ([
        ("author", "defend or revise the document under review",
         "You quote the passage you are changing and say what you changed."),
        ("fact checker", "verify every quantitative claim",
         "You check each number against the evidence cited and flag the "
         "unsupported ones."),
        ("structure reviewer", "judge whether the argument holds",
         "You quote the passages doing the load-bearing work and say whether "
         "they support the conclusion."),
        ("adversarial reviewer", "argue the central claim is wrong",
         "You construct the strongest case against the claim, quoting the "
         "text you attack."),
        ("methods auditor", "audit what was measured and what was assumed",
         "You identify conclusions the measurements cannot support."),
        ("related work reviewer", "place the work against prior art",
         "You identify novelty claims that prior work already covers."),
        ("editor", "maintain the required-revision list",
         "You keep every revision tied to the exact passage it applies to."),
    ],
        "Review this claim and the argument for it: LRU is the wrong eviction "
        "policy for LLM agent teams, because team members take turns and the "
        "member silent longest is the one about to speak, so LRU evicts "
        "exactly the session about to be needed; therefore serving systems "
        "should partition cache capacity per team."),
}


def get_crew(name, members):
    if name not in CREW:
        raise SystemExit(f"unknown task {name!r}; have {sorted(CREW)}")
    roles, goal = CREW[name]
    if members > len(roles):
        raise SystemExit(f"task {name!r} defines only {len(roles)} roles")
    return roles[:members], goal
