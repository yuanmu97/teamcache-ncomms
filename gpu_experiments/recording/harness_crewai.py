#!/usr/bin/env python3
"""Drive a CrewAI sequential crew against our endpoint, through the proxy.

The AutoGen companion (`harness_roundrobin.py`) serialises speaking with an
explicit round-robin selector. CrewAI reaches the same place by a different
route: a sequential `Process` runs tasks in declaration order and feeds each
task's output forward as context. Two harnesses that arrive at a ring by
unrelated designs is a stronger claim that the ring is a deployment reality
than either one alone, so both are recorded.

The rotation is written out explicitly: tasks cycle through the agents for
`--rounds` passes, which is the "hard rotation" configuration rather than
anything CrewAI infers.

Run with the recording proxy in front:

  python3 proxy_openai.py --trace traces/crewai-ring-k5.jsonl --listen-port 8901 &
  .venv-crewai/bin/python harness_crewai.py --members 5 --rounds 4
"""

from __future__ import annotations

import argparse
import os
import random

import tasks as taskbank

from crewai import Agent, Crew, LLM, Process, Task


def preamble(team: str, member: str, approx_tokens: int, seed: int = 0) -> str:
    if approx_tokens <= 0:
        return ""
    rng = random.Random(f"{seed}:{team}:{member}")
    pkgs = ["engine", "router", "memcache", "sched", "proto", "telemetry",
            "admission", "runtime", "codec", "store"]
    kinds = ["Handle", "Table", "Budget", "Window", "Cursor", "Shard",
             "Ticket", "Lease", "Digest", "Frame"]
    verbs = ["acquire", "reconcile", "drain", "promote", "seal", "rebind",
             "quiesce", "compact", "arm", "settle"]
    head = ("\n\nModules you own in this repository (do not modify others'):\n")
    lines, chars = [], len(head)
    while chars < approx_tokens * 3.17:
        line = (f"  - src/{rng.choice(pkgs)}/{rng.choice(verbs)}_"
                f"{rng.randrange(1000, 9999)}.py:"
                f"{rng.randrange(10, 990)}  "
                f"{rng.choice(verbs)}_{rng.choice(pkgs)}("
                f"{rng.choice(kinds).lower()}: {rng.choice(kinds)}) -> "
                f"{rng.choice(kinds)}  # owner={team}/{member} "
                f"rev={rng.randrange(10**6, 10**7)}\n")
        lines.append(line)
        chars += len(line)
    return head + "".join(lines)

# Roles and goal come from the shared bank (`tasks.py`) in CrewAI's own
# (role, goal, backstory) shape, so that a comparison across harnesses, tasks
# and models is the same ruler. The `design` class is byte-identical to what
# this file used to define inline.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8901/v1")
    ap.add_argument("--model", default="/opt/dlami/nvme/models/Kimi-K3")
    ap.add_argument("--task", default="design", choices=sorted(taskbank.CREW))
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--team", default="t0",
                    help="team id the gateway stamps into request ids")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--preamble-tokens", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()
    roles, GOAL = taskbank.get_crew(cfg.task, cfg.members)

    os.environ.setdefault("OPENAI_API_KEY", "none")
    # As in the AutoGen harness: name the team, say nothing about policy.
    llm_kw = dict(model=f"openai/{cfg.model}", base_url=cfg.base_url,
                  api_key="none", timeout=1800,
                  extra_headers={"x-team": cfg.team})
    if cfg.max_tokens:
        llm_kw["max_tokens"] = cfg.max_tokens
    llm = LLM(**llm_kw)

    agents = []
    for role, goal, story in roles:
        extra = preamble(cfg.team, role, cfg.preamble_tokens, cfg.seed)
        agents.append(Agent(
            role=role, goal=f"{GOAL} Your part: {goal}",
            backstory=story + extra, llm=llm, verbose=False,
            allow_delegation=False))
    # Declaration order is execution order under Process.sequential, so
    # cycling the agent list here is what makes the speaking order a ring.
    tasks = []
    for rnd in range(cfg.rounds):
        for agent in agents:
            tasks.append(Task(
                description=(
                    f"Round {rnd + 1}. {GOAL} Contribute substantive new "
                    f"material from your role, building on everything said "
                    f"so far. Be concrete and quantitative. Do not defer to "
                    f"the others and do not declare the work finished."),
                expected_output="A detailed, concrete contribution.",
                agent=agent,
                context=tasks[-len(agents):] if tasks else [],
            ))

    crew = Crew(agents=agents, tasks=tasks, process=Process.sequential,
                verbose=False, memory=False)
    result = crew.kickoff()
    print(f"\ntasks: {len(tasks)}  output chars: {len(str(result))}")


if __name__ == "__main__":
    main()
