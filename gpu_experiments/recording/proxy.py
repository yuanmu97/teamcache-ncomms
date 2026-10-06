#!/usr/bin/env python3
"""Instrumented Anthropic-Messages proxy for peer-agent-team serving research.

Sits between an agent harness (Claude Code Agent Teams) and a self-hosted
vLLM/SGLang endpoint that natively serves /v1/messages.

Three jobs:

1. Compatibility. Claude Code appends a ``[1m]`` context-beta suffix to model
   names it does not recognise; upstream rejects that with 404. Also optionally
   injects ``reasoning_effort`` so Kimi K3 stops emitting its chain of thought
   as the answer.

2. Protocol-level tracing. Records one JSONL row per request with arrival time,
   per-message sizes, and a cumulative prefix hash chain taken at message
   boundaries. The hash chain is what lets us measure, offline, where a
   teammate's output lands inside each member's prompt -- i.e. whether shared
   content is reusable as a prefix or not.

3. Peer-wait observation. Records ``stop_reason`` per request so we can count
   how often a member's turn ends with a plain ``end_turn`` while it is in fact
   waiting on a teammate -- the protocol ambiguity the research targets.

stdlib only; runs on the stock python3 of the machine driving the harness.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Headers we must not forward verbatim to upstream.
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

# Headers worth keeping in the trace. Authorization is never recorded.
_TRACED_HEADERS = (
    "user-agent",
    "x-app",
    "anthropic-version",
    "anthropic-beta",
    "x-stainless-helper-method",
)


def _text_of(block) -> str:
    """Flatten one Anthropic content block (or a bare string) to text."""
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    kind = block.get("type")
    if kind == "text":
        return block.get("text", "") or ""
    if kind == "thinking":
        return block.get("thinking", "") or ""
    if kind == "tool_use":
        return json.dumps(block.get("input", {}), sort_keys=True, ensure_ascii=False)
    if kind == "tool_result":
        content = block.get("content")
        if isinstance(content, list):
            return "".join(_text_of(b) for b in content)
        return content if isinstance(content, str) else ""
    return ""


def _flatten(content) -> str:
    if isinstance(content, list):
        return "".join(_text_of(b) for b in content)
    return _text_of(content)


def _block_kinds(content) -> list:
    if isinstance(content, list):
        return [b.get("type") for b in content if isinstance(b, dict)]
    return ["text"] if content else []


class Tracer:
    """Append-only JSONL writer, safe across handler threads."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", buffering=1, encoding="utf-8")

    def write(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def describe_prompt(body: dict) -> dict:
    """Summarise a Messages request into the shape the analysis needs.

    The prefix hash chain is cumulative: ``prefix_sha[i]`` covers the system
    prompt plus messages ``[0, i]``. Comparing chains across two requests gives
    the length of their common prompt prefix in messages, which is exactly the
    quantity a radix prefix cache can exploit.
    """
    system = body.get("system")
    system_text = _flatten(system) if system is not None else ""

    hasher = hashlib.sha1()
    hasher.update(system_text.encode("utf-8"))
    prefix_sha = [hasher.hexdigest()]

    roles, chars, kinds = [], [], []
    for msg in body.get("messages", []) or []:
        text = _flatten(msg.get("content"))
        roles.append(msg.get("role"))
        chars.append(len(text))
        kinds.append(_block_kinds(msg.get("content")))
        hasher.update(b"\x00")
        hasher.update((msg.get("role") or "").encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(text.encode("utf-8"))
        prefix_sha.append(hasher.hexdigest())

    tools = body.get("tools") or []
    return {
        "system_chars": len(system_text),
        "n_messages": len(roles),
        "msg_roles": roles,
        "msg_chars": chars,
        "msg_block_kinds": kinds,
        "prompt_chars": len(system_text) + sum(chars),
        "prefix_sha": prefix_sha,
        "n_tools": len(tools),
        "tool_names": [t.get("name") for t in tools if isinstance(t, dict)],
        "stream": bool(body.get("stream")),
        "max_tokens": body.get("max_tokens"),
        "metadata": body.get("metadata"),
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "pat-proxy/1"

    # Populated by main().
    cfg = None
    tracer = None
    _seq_lock = threading.Lock()
    _seq = 0

    def log_message(self, fmt, *args):  # quieter default logging
        if self.cfg and self.cfg.verbose:
            sys.stderr.write("[proxy] " + (fmt % args) + "\n")

    @classmethod
    def _next_seq(cls) -> int:
        with cls._seq_lock:
            cls._seq += 1
            return cls._seq

    def _upstream(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(
            self.cfg.upstream_host, self.cfg.upstream_port, timeout=self.cfg.timeout
        )

    def do_GET(self):
        self._relay(b"")

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        # Claude Code posts to /v1/messages?beta=true, so match on the path only.
        route = self.path.split("?", 1)[0].rstrip("/")
        if route.endswith("/v1/messages"):
            self._handle_messages(raw)
        else:
            self._relay(raw)

    # ---- generic pass-through (models, count_tokens, health) ----

    def _relay(self, raw: bytes):
        try:
            conn = self._upstream()
            conn.request(self.command, self.path, body=raw, headers=self._fwd_headers(raw))
            resp = conn.getresponse()
            payload = resp.read()
        except Exception as exc:  # upstream unreachable
            self.send_error(502, f"upstream error: {exc}")
            return
        self._send_headers(resp, extra={"content-length": str(len(payload))})
        self.wfile.write(payload)
        self.wfile.flush()

    def _fwd_headers(self, raw: bytes) -> dict:
        out = {}
        for key, value in self.headers.items():
            if key.lower() in _HOP_BY_HOP:
                continue
            out[key] = value
        out["content-length"] = str(len(raw))
        return out

    # ---- the interesting path ----

    def _handle_messages(self, raw: bytes):
        seq = self._next_seq()
        t_recv = time.time()
        mono = time.monotonic()

        try:
            body = json.loads(raw or b"{}")
        except Exception:
            self.send_error(400, "malformed JSON body")
            return

        model_in = body.get("model")
        model_out = self._rewrite_model(model_in)
        if model_out != model_in:
            body["model"] = model_out
        if self.cfg.reasoning_effort:
            body.setdefault("reasoning_effort", self.cfg.reasoning_effort)

        summary = describe_prompt(body)
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")

        row = {
            "seq": seq,
            "t_recv": t_recv,
            "model_in": model_in,
            "model_out": model_out,
            "headers": {
                k: self.headers.get(k) for k in _TRACED_HEADERS if self.headers.get(k)
            },
        }
        row.update(summary)

        try:
            conn = self._upstream()
            conn.request("POST", self.path, body=payload, headers=self._fwd_headers(payload))
            resp = conn.getresponse()
        except Exception as exc:
            row.update({"error": f"upstream: {exc}", "t_done": time.time()})
            self.tracer.write(row)
            self.send_error(502, f"upstream error: {exc}")
            return

        row["status"] = resp.status
        is_stream = "text/event-stream" in (resp.getheader("content-type") or "")

        if is_stream:
            self._send_headers(resp, extra={"transfer-encoding": "chunked"})
            self._pump_stream(resp, row, mono)
        else:
            # Upstream may answer chunked. We strip transfer-encoding as a
            # hop-by-hop header, so we have to declare our own length or the
            # client waits for an EOF that keep-alive never delivers.
            data = resp.read()
            self._send_headers(resp, extra={"content-length": str(len(data))})
            try:
                self.wfile.write(data)
                self.wfile.flush()
            except Exception:
                pass
            row["t_first_byte"] = row.get("t_first_byte") or time.time()
            self._record_nonstream(data, row)

        row["t_done"] = time.time()
        row["wall_s"] = time.monotonic() - mono
        self.tracer.write(row)

    def _send_headers(self, resp, extra=None):
        """Copy upstream headers, minus hop-by-hop, plus our own framing."""
        extra = extra or {}
        self.send_response(resp.status)
        skip = set(_HOP_BY_HOP) | {k.lower() for k in extra}
        for key, value in resp.getheaders():
            if key.lower() in skip:
                continue
            self.send_header(key, value)
        for key, value in extra.items():
            self.send_header(key, value)
        self.end_headers()

    def _rewrite_model(self, model):
        """Strip Claude Code's ``[...]`` context-beta suffix, then apply aliases."""
        if not isinstance(model, str):
            return model
        name = model
        if name.endswith("]") and "[" in name:
            name = name[: name.rindex("[")]
        return self.cfg.alias.get(name, self.cfg.alias.get("*", name))

    def _record_nonstream(self, data: bytes, row: dict):
        try:
            obj = json.loads(data)
        except Exception:
            return
        row["stop_reason"] = obj.get("stop_reason")
        row["usage"] = obj.get("usage")
        row["out_block_kinds"] = _block_kinds(obj.get("content"))
        row["out_chars"] = len(_flatten(obj.get("content")))

    def _pump_stream(self, resp, row: dict, mono: float):
        """Relay SSE verbatim while harvesting usage and stop_reason."""
        first = True
        out_chars = 0
        kinds = []
        buf = b""
        while True:
            try:
                chunk = resp.read(4096)
            except Exception as exc:
                row["error"] = f"stream read: {exc}"
                break
            if not chunk:
                break
            if first:
                row["t_first_byte"] = time.time()
                row["ttfb_s"] = time.monotonic() - mono
                first = False
            try:
                self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                self.wfile.flush()
            except Exception:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except Exception:
                    continue
                etype = event.get("type")
                if etype == "content_block_start":
                    block = event.get("content_block") or {}
                    kinds.append(block.get("type"))
                elif etype == "content_block_delta":
                    delta = event.get("delta") or {}
                    out_chars += len(
                        delta.get("text") or delta.get("thinking") or delta.get("partial_json") or ""
                    )
                elif etype == "message_delta":
                    delta = event.get("delta") or {}
                    if delta.get("stop_reason"):
                        row["stop_reason"] = delta["stop_reason"]
                    if event.get("usage"):
                        row["usage"] = event["usage"]
                elif etype == "message_start":
                    msg = event.get("message") or {}
                    if msg.get("usage"):
                        row["usage_start"] = msg["usage"]
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass
        row["out_chars"] = out_chars
        row["out_block_kinds"] = kinds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listen-port", type=int, default=8900)
    ap.add_argument("--upstream-host", default="127.0.0.1")
    ap.add_argument("--upstream-port", type=int, default=8888)
    ap.add_argument("--trace", default="traces/proxy.jsonl")
    ap.add_argument(
        "--alias",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="rewrite a model name; use '*=NAME' to force every request to NAME",
    )
    ap.add_argument(
        "--reasoning-effort",
        default="",
        help="inject a top-level reasoning_effort (Kimi K3 wants this, not chat_template_kwargs)",
    )
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--verbose", action="store_true")
    cfg = ap.parse_args(argv)

    cfg.alias = dict(a.split("=", 1) for a in cfg.alias)
    trace_dir = os.path.dirname(os.path.abspath(cfg.trace))
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)

    Handler.cfg = cfg
    Handler.tracer = Tracer(cfg.trace)

    server = ThreadingHTTPServer(("127.0.0.1", cfg.listen_port), Handler)
    server.daemon_threads = True
    sys.stderr.write(
        f"[proxy] :{cfg.listen_port} -> {cfg.upstream_host}:{cfg.upstream_port} "
        f"trace={cfg.trace} alias={cfg.alias or '{}'}\n"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.tracer.close()


if __name__ == "__main__":
    main()
