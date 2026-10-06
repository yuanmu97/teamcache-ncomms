"""How to put team/member/turn on the request id SGLang sees.

The TeamCache predicate only acts when that id is ``team:member:turn`` and
``member`` is decimal digits. Anything else is left to upstream LRU.

Installing this package does not rewrite traffic that lacks a request id.
If you do not already have team/member metadata, infer them offline with
``teamcache.discovery`` and then set the rid here.
"""


def rid(team: str, member: int, turn: int) -> str:
    if not str(member).isdigit():
        raise ValueError("member must be decimal digits")
    return f"{team}:{int(member)}:{int(turn)}"


# Native SGLang /generate:
#
#   POST /generate  {"input_ids": [...], "rid": "alpha:0:3", ...}
#
# OpenAI-compatible server: SGLang reads body["rid"]. Also set the request-id
# header if a proxy sits in front.
#
#   from openai import OpenAI
#   client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")
#   client.chat.completions.create(
#       model="default",
#       messages=[{"role": "user", "content": "..."}],
#       extra_body={"rid": rid("alpha", 0, 3)},
#   )
