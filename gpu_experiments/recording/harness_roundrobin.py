#!/usr/bin/env python3
"""Drive AutoGen's RoundRobinGroupChat against our endpoint, through the proxy.

Why this exists. The measured Claude Code traces are not a ring (mean recency
rank 0.79 at k=4 and 1.77 at k=7, against a ring's 3 and 6), so the ring had
to be demoted from "the shape we observe" to "an analytical worst case". But
it is also a *deployment configuration*: AutoGen's RoundRobinGroupChat and
CrewAI's sequential process serialise speaking by construction. If that is
true on the wire and not just in the documentation, the ring is a real regime
and the mechanism has a real population to serve. This script is how we find
out, rather than asserting it.

The task is deliberately one that keeps every member writing long turns, so
each member accumulates the private history that makes residency matter. A
task that collapses into one agent doing the work would measure nothing.

Run with the recording proxy in front:

  python3 proxy_openai.py --trace traces/autogen-rr1.jsonl --listen-port 8900 &
  .venv-harness/bin/python harness_roundrobin.py --members 5 --rounds 8
"""

from __future__ import annotations

import argparse
import asyncio
import random

import tasks


def preamble(team: str, member: str, approx_tokens: int, seed: int = 0) -> str:
    """A private, deterministic block of context for one member of one team.

    Why the harness needs this. Started from an empty context, every member's
    first prompt is ~380 tokens, so for the first half of a run the whole
    fleet fits in the pool and LRU hits 84% without any mechanism. That warm
    phase is an artefact of the harness, not of the workload: a real member's
    first prompt already carries its system prompt, its tool definitions and
    the slice of the repository it owns -- which is why the measured Claude
    Code contexts have a *median* of 9,728 tokens rather than a ramp starting
    near zero. Measured consequence of the artefact: on the 13-round AutoGen
    ring trace the retention rule reads its own p_hit as 0.55 (averaged across
    the fits/does-not-fit transition) and never engages, while on the same
    trace restricted to the contended rounds it engages and gains +24.3pp.

    The block is keyed to the *team as well as* the member, and that is not a
    detail. Seeding it by role alone made all 20 teams' Architects share one
    preamble, so the fleet had 5 distinct prefixes instead of 100; they fit the
    pool trivially, were touched by 20 teams per round, and never evicted. LRU
    then measured 79.5% on what was supposed to be a contended ring, and the
    offline replayer -- which assumed private contexts -- predicted 14.0%. The
    trace says it outright: one distinct `prefix_sha` per member id across all
    20 teams. Independent teams work on different problems, so the block has
    to be private per session; only the ~200-token role prompt stays shared,
    which is realistic for a fleet running one harness and too small to matter.

    It is placed after that member's own system prompt, so no two members of a
    team share a prefix either -- which is what the collected traces show
    (0 shared messages between members of a team).
    """
    if approx_tokens <= 0:
        return ""
    rng = random.Random(f"{seed}:{team}:{member}")
    pkgs = ["engine", "router", "memcache", "sched", "proto", "telemetry",
            "admission", "runtime", "codec", "store"]
    kinds = ["Handle", "Table", "Budget", "Window", "Cursor", "Shard",
             "Ticket", "Lease", "Digest", "Frame"]
    verbs = ["acquire", "reconcile", "drain", "promote", "seal", "rebind",
             "quiesce", "compact", "arm", "settle"]
    head = (f"\n\nModules you own in this repository (do not modify others'):"
            f"\n")
    lines, chars = [], len(head)
    # 3.17 chars per token, measured against the served tokenizer by asking
    # the engine itself for `prompt_tokens` on this exact text at 2K, 6K and
    # 9K (coefficients 1.260 / 1.261 / 1.264 against a 4-char estimate).
    # Dense identifier text tokenises well below prose, and guessing 4 here
    # would oversize every context by 26% -- the same class of silent error
    # that cost two end-to-end runs, so it is calibrated rather than assumed.
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

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Roles and task come from the shared bank (`tasks.py`) so that a comparison
# across harnesses, tasks and models is the same ruler. The `design` class is
# byte-identical to what this file used to define inline, which is what keeps
# the already-collected k=5 ring traces comparable to everything collected
# after it.


async def run(cfg):
    client = OpenAIChatCompletionClient(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key="none",
        timeout=1800,
        # The only line in the harness that knows about the serving mechanism,
        # and it names the team rather than describing a policy. The gateway
        # turns it into a per-request id; AutoGen is otherwise untouched.
        default_headers={"x-team": cfg.team},
        max_tokens=cfg.max_tokens or None,
        model_info={
            "vision": False, "function_calling": False, "json_output": False,
            "family": "unknown", "structured_output": False,
        },
    )
    roles, task = tasks.get(cfg.task, cfg.members)
    agents = [
        AssistantAgent(name=name, model_client=client,
                       system_message=prompt + preamble(
                           cfg.team, name, cfg.preamble_tokens, cfg.seed))
        for name, prompt in roles
    ]
    # RoundRobinGroupChat is the point: it picks speakers by position, not by
    # a selector model, so the speaking order is fixed by the harness and not
    # by anything the task or the model decides.
    team = RoundRobinGroupChat(
        agents,
        termination_condition=MaxMessageTermination(
            cfg.members * cfg.rounds + 1),
    )
    result = await team.run(task=task)
    print(f"\nmessages: {len(result.messages)}  stop: {result.stop_reason}")
    for msg in result.messages:
        src = getattr(msg, "source", "?")
        content = getattr(msg, "content", "")
        print(f"  {src:<12} {len(str(content)):>7} chars")
    await client.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8900/v1")
    ap.add_argument("--model", default="/opt/dlami/nvme/models/Kimi-K3")
    ap.add_argument("--task", default="design", choices=sorted(tasks.TASKS))
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="cap each turn; 0 lets the model decide. The "
                         "collection runs uncapped so the measured length "
                         "distribution is the model's own; the end-to-end "
                         "arms cap it so every arm decodes the same amount "
                         "and the comparison is about residency, not luck.")
    ap.add_argument("--preamble-tokens", type=int, default=0,
                    help="private context to give each member up front, so "
                         "the run is contended from its first turn instead "
                         "of spending its first half inside the pool. 0 "
                         "reproduces the earlier collection runs.")
    ap.add_argument("--team", default="t0",
                    help="team id the gateway stamps into request ids")
    ap.add_argument("--seed", type=int, default=0,
                    help="harness seed for the private preamble; change this "
                         "to get a new prefix without changing geometry")
    cfg = ap.parse_args()
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
