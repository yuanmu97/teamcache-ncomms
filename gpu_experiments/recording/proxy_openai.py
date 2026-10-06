#!/usr/bin/env python3
"""Instrumented OpenAI chat-completions proxy for peer-agent-team traces.

`proxy.py` records Claude Code, which speaks Anthropic Messages. The harnesses
that serialise speaking -- AutoGen's RoundRobinGroupChat, CrewAI's sequential
process -- speak OpenAI chat/completions instead, and those are the ones whose
speaking order we need on record. Same three jobs as `proxy.py`, same JSONL
row shape where the fields mean the same thing, so `analyze_trace.py` reads
either without knowing which harness produced it.

Recovering *which member* sent a request is the whole point and it cannot be
taken from the wire: neither harness labels the caller, and both reuse one
connection and one model name for every agent. So sessions are recovered the
same way `analyze_trace.py` does it for Claude Code -- by chaining cumulative
prefix hashes -- and this proxy's job is only to record the chain honestly.

stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}

_TRACED_HEADERS = ("user-agent", "x-stainless-helper-method", "x-harness",
                   "x-team")


def _flatten(content) -> str:
    """Flatten one OpenAI message content to text.

    Content is either a string or, for the multimodal shape, a list of parts.
    Tool calls live beside content rather than inside it, so they are folded in
    by the caller.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                out.append(part.get("text") or "")
        return "".join(out)
    return ""


def _message_text(msg: dict) -> str:
    text = _flatten(msg.get("content"))
    calls = msg.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            fn = (call or {}).get("function") or {}
            text += fn.get("name", "") + (fn.get("arguments") or "")
    return text


class Tracer:
    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._fh = open(path, "a", buffering=1, encoding="utf-8")

    def write(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._fh.write(line + "\n")


def describe_prompt(body: dict) -> dict:
    """Summarise a chat/completions request into the analysis shape.

    `prefix_sha[i]` covers messages [0, i] inclusive, so the longest common
    prefix of two requests' chains is their shared prompt prefix in messages --
    exactly what a radix cache can reuse, and what session recovery keys on.
    """
    messages = body.get("messages") or []
    running = hashlib.sha256()
    sizes, roles, chain = [], [], []
    for msg in messages:
        text = _message_text(msg if isinstance(msg, dict) else {})
        role = (msg or {}).get("role", "?")
        running.update(role.encode("utf-8", "replace"))
        running.update(b"\x00")
        running.update(text.encode("utf-8", "replace"))
        running.update(b"\x01")
        sizes.append(len(text))
        roles.append(role)
        chain.append(running.hexdigest())
    return {
        "model_in": body.get("model"),
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens"),
        "msg_roles": roles,
        "msg_sizes": sizes,
        "prefix_sha": chain,
        "n_messages": len(messages),
        "tool_names": [((t or {}).get("function") or {}).get("name")
                       for t in (body.get("tools") or [])],
    }


class RidMinter:
    """Name requests `<team>:<member>:<turn>`, which is all the mechanism needs.

    The engine-side policy reads group and member off the request id and takes
    the untouched upstream path for ids without that shape, so this is the
    entire client-side cost of running a real harness under Cohort. Neither
    AutoGen nor CrewAI has any notion of it, and neither has to: the member
    index is recovered here from the system prompt, which both harnesses make
    unique per agent, and the team comes from a header the launcher sets.

    That is the honest version of "a naming convention is the protocol
    change" -- it is done by the gateway, without touching the harness.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._members = {}   # team -> {system-prompt hash: member index}
        self._turns = {}     # (team, member) -> next turn number

    def mint(self, team, sys_hash):
        with self._lock:
            table = self._members.setdefault(team, {})
            member = table.setdefault(sys_hash, len(table))
            key = (team, member)
            turn = self._turns.get(key, 0)
            self._turns[key] = turn + 1
        return f"{team}:{member}:{turn}", member


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "pat-openai-proxy"

    def log_message(self, fmt, *args):  # quieter than the default
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        t_recv = time.time()
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            body = {}

        row = {"t_recv": t_recv, "path": self.path}
        row.update(describe_prompt(body))
        row["headers"] = {h: self.headers.get(h) for h in _TRACED_HEADERS
                          if self.headers.get(h)}

        # Streaming is left alone: the harnesses do not need it and folding a
        # stream back together would put reassembly bugs inside the trace.
        body.pop("stream", None)
        body.pop("stream_options", None)
        # Kimi K3 otherwise returns its chain of thought in `reasoning_content`
        # and leaves `content` empty, which a harness reads as an agent that
        # said nothing -- the round-robin would still rotate, but every turn
        # would carry no text and the trace would measure an empty task.
        if self.server.reasoning_effort:
            body.setdefault("reasoning_effort", self.server.reasoning_effort)
        # GLM-4.7-Flash spends a 144-token cap on thinking and returns no
        # speaker name to the selector; PROXY_CHAT_TEMPLATE_KWARGS switches
        # thinking off for such models.
        if os.environ.get("PROXY_CHAT_TEMPLATE_KWARGS"):
            body.setdefault("chat_template_kwargs",
                            json.loads(os.environ["PROXY_CHAT_TEMPLATE_KWARGS"]))
        if self.server.minter is not None and row.get("prefix_sha"):
            team = self.headers.get("x-team") or self.server.default_team
            rid, member = self.server.minter.mint(team, row["prefix_sha"][0])
            body["rid"] = rid
            row["rid"] = rid
            row["team"] = team
            row["member"] = member
        payload = json.dumps(body).encode("utf-8")

        conn = http.client.HTTPConnection(self.server.up_host,
                                          self.server.up_port, timeout=1800)
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in _HOP_BY_HOP}
        headers["content-length"] = str(len(payload))
        # SGLang reads `body["rid"]`. vLLM names the request from this header
        # (`request.request_id`); without it the cohort hook never sees a team.
        if row.get("rid"):
            headers["X-Request-Id"] = row["rid"]
            headers["X-Request-ID"] = row["rid"]
        try:
            conn.request("POST", self.path, body=payload, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            status = resp.status
        except Exception as exc:
            row["error"] = repr(exc)
            self.server.tracer.write(row)
            self.send_error(502, "upstream failed")
            return

        row["t_done"] = time.time()
        row["status"] = status
        try:
            doc = json.loads(data)
            choice = (doc.get("choices") or [{}])[0]
            usage = doc.get("usage") or {}
            row["model_out"] = doc.get("model")
            # finish_reason is the OpenAI analogue of stop_reason: the field
            # that cannot tell "the task is done" from "I am waiting on a
            # teammate", which is the protocol ambiguity under study.
            row["finish_reason"] = choice.get("finish_reason")
            msg = choice.get("message") or {}
            row["out_text_len"] = len(_message_text(msg))
            row["reasoning_len"] = len(msg.get("reasoning_content") or "")
            # If the model still answers only in reasoning_content, hand that
            # to the harness rather than an empty turn. Recording both lengths
            # keeps the substitution visible in the trace instead of silent.
            if not _message_text(msg) and msg.get("reasoning_content"):
                msg["content"] = msg["reasoning_content"]
                row["reasoning_substituted"] = True
                data = json.dumps(doc).encode("utf-8")
            row["prompt_tokens"] = usage.get("prompt_tokens")
            row["completion_tokens"] = usage.get("completion_tokens")
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            row["cached_tokens"] = cached
        except Exception:
            pass
        self.server.tracer.write(row)

        self.send_response(status)
        for key, value in resp.getheaders():
            if key.lower() in _HOP_BY_HOP:
                continue
            self.send_header(key, value)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.do_POST() if False else self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen-port", type=int, default=8900)
    ap.add_argument("--upstream-host", default="127.0.0.1")
    ap.add_argument("--upstream-port", type=int, default=8888)
    ap.add_argument("--trace", default="traces/openai.jsonl")
    # Unused except that it lands in the command line, which gives the
    # collection script a string it can pkill without also matching itself.
    ap.add_argument("--tag", default="")
    ap.add_argument("--inject-rid", action="store_true",
                    help="name requests <team>:<member>:<turn> so the engine-"
                         "side group policy can see teams in a real harness")
    ap.add_argument("--default-team", default="t0",
                    help="team id for requests without an x-team header")
    ap.add_argument("--reasoning-effort", default="low",
                    help="injected top-level; Kimi K3 wants this rather than "
                         "chat_template_kwargs")
    cfg = ap.parse_args()

    httpd = ThreadingHTTPServer(("127.0.0.1", cfg.listen_port), Handler)
    httpd.up_host = cfg.upstream_host
    httpd.up_port = cfg.upstream_port
    httpd.tracer = Tracer(cfg.trace)
    httpd.reasoning_effort = cfg.reasoning_effort
    httpd.minter = RidMinter() if cfg.inject_rid else None
    httpd.default_team = cfg.default_team
    print(f"listening 127.0.0.1:{cfg.listen_port} -> "
          f"{cfg.upstream_host}:{cfg.upstream_port}  trace={cfg.trace}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
