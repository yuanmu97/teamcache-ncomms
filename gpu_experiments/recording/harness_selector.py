#!/usr/bin/env python3
"""Drive AutoGen's SelectorGroupChat: a free-form population we can generate.

§3.4 measures two populations with opposite requirements. The serialised one is
cheap to study because we produce it ourselves (`harness_roundrobin.py`,
`harness_crewai.py`, both in their documented default configuration). The
free-form one is not: it rests on two Claude Code runs, and Claude Code access
is the one input to this project we cannot scale. That asymmetry is the real
content of limitation §12.8 -- the conclusion that "the speaking order is a
property of the harness" is drawn from n=2 on one side.

SelectorGroupChat closes it. It asks a model which member should speak next
instead of advancing a position counter, so the order is decided at runtime by
the conversation rather than by the harness, which is structurally the same
thing Claude Code's peer messaging does. That gives us a free-form population
on our own hardware, at any k, on any task, under any model -- so the claim can
be tested at a sample size that means something.

What this can and cannot establish, stated plainly: it makes the *free-form
regime* reproducible and lets us check whether the Claude Code numbers
(recency rank 0.79 and 1.77, 77% false completions, duty cycle 0.277) are
typical of free-form peer messaging or particular to that client. It does not
replace the Claude Code traces as the anchor for the workload class, because
the isolation proof in §1.3 -- the same task given to a single agent, duty
cycle 0.962 and 0% false completions -- is what defines the class at all.

Run behind the recording proxy, which is where the measurement happens:

  python3 proxy_openai.py --trace traces/sel-design-k5.jsonl --listen-port 8900 &
  .venv-ag/bin/python harness_selector.py --task design --members 5 --rounds 8
"""

from __future__ import annotations

import argparse
import asyncio

import tasks
from harness_roundrobin import preamble

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient


# Deliberately does not forbid the same speaker twice in a row. AutoGen's
# default would, and that default is exactly the statistic under measurement:
# the dominant mode on the Claude Code traces is a member speaking again
# immediately (48% of turns at k=4). A harness that structurally forbids the
# modal behaviour cannot be used to check whether that behaviour is real.
ALLOW_REPEATED = True


async def run(cfg):
    def make_client(headers=True):
        return OpenAIChatCompletionClient(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key="none",
            timeout=1800,
            # Only the member clients are stamped with the team. The selector
            # is infrastructure, not a team member: giving it a request id
            # in the same namespace would make the gateway mint it a member
            # index and the policy would budget cache for a session that holds
            # no history worth retaining.
            default_headers={"x-team": cfg.team} if headers else {},
            max_tokens=cfg.max_tokens or None,
            model_info={
                "vision": False, "function_calling": False,
                "json_output": False, "family": "unknown",
                "structured_output": False,
            },
        )

    member_client = make_client(headers=True)
    selector_client = make_client(headers=False)

    roles, task = tasks.get(cfg.task, cfg.members)
    agents = [
        AssistantAgent(name=name, model_client=member_client,
                       system_message=prompt + preamble(
                           cfg.team, name, cfg.preamble_tokens, cfg.seed))
        for name, prompt in roles
    ]
    team = SelectorGroupChat(
        agents,
        model_client=selector_client,
        termination_condition=MaxMessageTermination(
            cfg.members * cfg.rounds + 1),
        allow_repeated_speaker=ALLOW_REPEATED,
    )
    result = await team.run(task=task)

    print(f"\ntask={cfg.task} messages={len(result.messages)} "
          f"stop={result.stop_reason}")
    order = [getattr(m, "source", "?") for m in result.messages]
    print("order: " + " ".join(order))
    for msg in result.messages:
        print(f"  {getattr(msg, 'source', '?'):<14} "
              f"{len(str(getattr(msg, 'content', ''))):>7} chars")
    await member_client.close()
    await selector_client.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8900/v1")
    ap.add_argument("--model", default="/opt/dlami/nvme/models/Kimi-K3")
    ap.add_argument("--task", default="design", choices=sorted(tasks.TASKS))
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--preamble-tokens", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--team", default="t0")
    cfg = ap.parse_args()
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
