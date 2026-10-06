"""Cluster-wide discovery coordinator and OpenAI-compatible affinity gateway.

The coordinator receives hashes and lengths only; raw prompt text remains at
the gateway. All gateway replicas should use the same coordinator so member
turns and union-find decisions are cluster-wide. The gateway then uses
rendezvous hashing on the discovered team id to keep a team on one SGLang
backend while distributing unrelated teams across the fleet.
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import os
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from teamcache.discovery import TeamDiscovery

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "trailers", "transfer-encoding", "upgrade", "host",
    "content-length",
}


def _flatten(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out = []
    for part in content:
        if isinstance(part, str):
            out.append(part)
        elif isinstance(part, dict):
            out.append(part.get("text") or "")
    return "".join(out)


def _message_text(message) -> str:
    message = message if isinstance(message, dict) else {}
    text = _flatten(message.get("content"))
    for call in message.get("tool_calls") or []:
        fn = (call or {}).get("function") or {}
        text += str(fn.get("name") or "") + str(fn.get("arguments") or "")
    return text


def describe_request(body: dict, now=None, key=None) -> dict:
    """Build privacy-safe cumulative fingerprints for one OpenAI request."""
    if key:
        secret = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        running = hmac.new(secret, digestmod=hashlib.sha256)
    else:
        running = hashlib.sha256()
    chain, roles, sizes = [], [], []
    for message in body.get("messages") or []:
        message = message if isinstance(message, dict) else {}
        role = str(message.get("role") or "?")
        text = _message_text(message)
        running.update(role.encode("utf-8", "replace"))
        running.update(b"\x00")
        running.update(text.encode("utf-8", "replace"))
        running.update(b"\x01")
        chain.append(running.hexdigest())
        roles.append(role)
        sizes.append(len(text))
    return {
        "t_recv": time.time() if now is None else float(now),
        "model_in": body.get("model") or "_",
        "prefix_sha": chain,
        "msg_roles": roles,
        "msg_sizes": sizes,
        "n_messages": len(chain),
    }


def response_text_len(document: dict) -> int:
    total = 0
    for choice in document.get("choices") or []:
        message = (choice or {}).get("message") or {}
        total += len(_message_text(message))
        total += len(message.get("reasoning_content") or "")
    return total


def team_from_rid(rid):
    """Extract a team identifier from a strict ``team:member:turn`` id."""
    if not isinstance(rid, str):
        return None
    parts = rid.split(":")
    if len(parts) != 3 or not parts[0] or not parts[1].isdigit() or not parts[2].isdigit():
        return None
    return parts[0]


class DiscoveryCoordinator:
    """Thread-safe cluster authority for identity, turns, and completion."""

    def __init__(self, discovery=None, max_requests=100000, namespace=None):
        self.discovery = discovery or TeamDiscovery()
        self.max_requests = int(max_requests)
        self.namespace = str(namespace or f"e{uuid.uuid4().hex[:8]}")
        if not self.namespace or ":" in self.namespace:
            raise ValueError("discovery namespace must be non-empty and contain no colon")
        self._lock = threading.RLock()
        self._pending = OrderedDict()
        self._known = OrderedDict()
        self._contexts = {}
        self._metrics = defaultdict(int)

    def identify(self, row, tenant="_", model="_", request_key=None):
        with self._lock:
            tenant = tenant or "_"
            model = model or "_"
            scoped_request_key = (
                (tenant, model, str(request_key)) if request_key else None
            )
            if scoped_request_key and scoped_request_key in self._known:
                self._metrics["idempotent_identify"] += 1
                result = dict(self._known[scoped_request_key])
                if result["completion_id"] not in self._pending:
                    completion_id = uuid.uuid4().hex
                    result["completion_id"] = completion_id
                    self._known[scoped_request_key] = dict(result)
                    context = self._contexts.get(scoped_request_key)
                    if context is not None:
                        self._pending[completion_id] = context
                self._known.move_to_end(scoped_request_key)
                return result
            # One coordinator owns expiry, so it must also own time. Gateway
            # wall clocks can be skewed (or supplied by an untrusted caller),
            # while monotonic time is stable for the coordinator lifetime.
            observed = dict(row)
            observed["_t"] = time.monotonic()
            ident, ticket = self.discovery.begin(observed, tenant=tenant, model=model)
            member = int(ident.member_id)
            turn = self.discovery.turn_of(member)
            completion_id = uuid.uuid4().hex
            team_id = f"{self.namespace}-{self.discovery.team_of(member)}"
            result = {
                "team_id": team_id,
                "member_id": member,
                "turn": turn,
                "rid": f"{team_id}:{member}:{turn}",
                "completion_id": completion_id,
                "lineage_new": bool(ident.lineage_new),
                "handoffs": int(ident.n_handoffs),
            }
            self._pending[completion_id] = (member, ticket)
            if scoped_request_key:
                self._known[scoped_request_key] = dict(result)
                self._contexts[scoped_request_key] = (member, ticket)
            self._trim()
            self._metrics["identified"] += 1
            self._metrics["new_lineages"] += int(ident.lineage_new)
            self._metrics["handoffs"] += int(ident.n_handoffs)
            return result

    def complete(self, completion_id, out_text_len, success=True):
        with self._lock:
            pending = self._pending.pop(str(completion_id), None)
            if pending is None:
                self._metrics["unknown_completion"] += 1
                return False
            member, ticket = pending
            if not success:
                self._metrics["aborted"] += 1
                return True
            ok = self.discovery.record_output(member, out_text_len, ticket)
            self._metrics["completed"] += int(ok)
            return ok

    def _trim(self):
        while len(self._pending) > self.max_requests:
            self._pending.popitem(last=False)
            self._metrics["pending_evicted"] += 1
        while len(self._known) > self.max_requests:
            scoped_request_key, _ = self._known.popitem(last=False)
            self._contexts.pop(scoped_request_key, None)

    def stats(self):
        with self._lock:
            return {
                **dict(self._metrics),
                "pending": len(self._pending),
                "remembered_requests": len(self._known),
                "namespace": self.namespace,
                "discovery": self.discovery.stats(),
            }


class LocalDiscoveryClient:
    """In-process discovery client for a single gateway deployment."""

    def __init__(self, coordinator=None):
        self.coordinator = coordinator or DiscoveryCoordinator()

    def identify(self, row, tenant, model, request_key):
        return self.coordinator.identify(row, tenant, model, request_key)

    def complete(self, completion_id, out_text_len, success=True):
        return self.coordinator.complete(completion_id, out_text_len, success)


class RemoteDiscoveryClient:
    """Persistent HTTP client for a shared discovery coordinator."""

    def __init__(self, base_url, token=None, timeout=5.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = float(timeout)
        self._parsed = urlsplit(self.base_url)
        if self._parsed.scheme not in ("http", "https") or not self._parsed.hostname:
            raise ValueError(f"invalid discovery URL: {base_url}")
        self._local = threading.local()

    def _connection(self):
        connection = getattr(self._local, "connection", None)
        if connection is None:
            cls = (
                http.client.HTTPSConnection
                if self._parsed.scheme == "https"
                else http.client.HTTPConnection
            )
            connection = cls(
                self._parsed.hostname, self._parsed.port, timeout=self.timeout
            )
            self._local.connection = connection
        return connection

    def _drop_connection(self):
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            try:
                connection.close()
            finally:
                self._local.connection = None

    def _post(self, path, body):
        headers = {"content-type": "application/json"}
        if self.token:
            headers["authorization"] = f"Bearer {self.token}"
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        target = self._parsed.path.rstrip("/") + path
        for attempt in range(2):
            try:
                connection = self._connection()
                connection.request("POST", target, body=payload, headers=headers)
                response = connection.getresponse()
                data = response.read()
                if response.status // 100 != 2:
                    raise RuntimeError(
                        f"discovery returned HTTP {response.status}: "
                        f"{data[:200].decode('utf-8', 'replace')}"
                    )
                return json.loads(data.decode("utf-8"))
            except (OSError, http.client.HTTPException):
                self._drop_connection()
                if attempt:
                    raise
        raise RuntimeError("unreachable")

    def identify(self, row, tenant, model, request_key):
        return self._post("/v1/identify", {
            "row": row,
            "tenant": tenant,
            "model": model,
            "request_key": request_key,
        })

    def complete(self, completion_id, out_text_len, success=True):
        result = self._post("/v1/complete", {
            "completion_id": completion_id,
            "out_text_len": int(out_text_len),
            "success": bool(success),
        })
        return bool(result.get("ok"))


class RendezvousRouter:
    """Minimal-disruption team affinity over an unordered backend set."""

    def __init__(self, backends):
        normalized = sorted({str(item).rstrip("/") for item in backends if item})
        if not normalized:
            raise ValueError("at least one backend is required")
        for backend in normalized:
            parsed = urlsplit(backend)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError(f"invalid backend URL: {backend}")
        self.backends = tuple(normalized)
        self._healthy = set(self.backends)
        self._lock = threading.RLock()

    @staticmethod
    def _score(key, backend):
        digest = hashlib.blake2b(
            f"{key}\x00{backend}".encode(), digest_size=16
        ).digest()
        return int.from_bytes(digest, "big")

    def ranked(self, key):
        key = str(key or "_")
        with self._lock:
            candidates = tuple(sorted(self._healthy)) or self.backends
        return sorted(candidates, key=lambda b: self._score(key, b), reverse=True)

    def choose(self, key):
        return self.ranked(key)[0]

    def mark_health(self, backend, healthy):
        if backend not in self.backends:
            raise KeyError(backend)
        with self._lock:
            if healthy:
                self._healthy.add(backend)
            else:
                self._healthy.discard(backend)

    def health(self):
        with self._lock:
            return {backend: backend in self._healthy for backend in self.backends}


class _SSECounter:
    def __init__(self):
        self.buffer = b""
        self.total = 0

    def feed(self, chunk):
        self.buffer += chunk
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                document = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                continue
            for choice in document.get("choices") or []:
                delta = (choice or {}).get("delta") or {}
                self.total += len(_message_text(delta))
                self.total += len(delta.get("reasoning_content") or "")


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 256


class DiscoveryHandler(BaseHTTPRequestHandler):
    """HTTP handler exposing identify, complete, health, and stats endpoints."""

    protocol_version = "HTTP/1.1"
    server_version = "teamcache-discovery"

    def log_message(self, fmt, *args):
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _authorized(self):
        token = self.server.token
        return not token or self.headers.get("authorization") == f"Bearer {token}"

    def _json(self, status, document):
        data = json.dumps(document, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
        elif self.path == "/health":
            self._json(200, {"ok": True})
        elif self.path == "/stats":
            self._json(200, self.server.coordinator.stats())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        length = int(self.headers.get("content-length") or 0)
        if length < 0 or length > self.server.max_body:
            self._json(413, {"error": "request too large"})
            return
        try:
            document = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/v1/identify":
                result = self.server.coordinator.identify(
                    document.get("row") or {},
                    tenant=document.get("tenant") or "_",
                    model=document.get("model") or "_",
                    request_key=document.get("request_key"),
                )
                self._json(200, result)
            elif self.path == "/v1/complete":
                ok = self.server.coordinator.complete(
                    document.get("completion_id"),
                    document.get("out_text_len") or 0,
                    document.get("success", True),
                )
                self._json(200, {"ok": ok})
            else:
                self._json(404, {"error": "not found"})
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})


class GatewayHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible proxy with discovery and team-affinity routing."""

    protocol_version = "HTTP/1.1"
    server_version = "teamcache-gateway"

    def log_message(self, fmt, *args):
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _error(self, status, message):
        data = json.dumps({"error": {"message": message, "type": "teamcache_gateway"}}).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            data = b'{"ok":true}'
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._proxy_without_discovery()

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        if length < 0 or length > self.server.max_body:
            self._error(413, "request too large")
            return
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._error(400, "request body must be JSON")
            return

        request_key = self.headers.get("x-request-id") or uuid.uuid4().hex
        completion_id = None
        team_key = None
        discovery_status = "explicit"
        metadata = body.pop("teamcache", None)
        rid = body.get("rid")
        if isinstance(metadata, dict) and metadata.get("team") is not None:
            try:
                team = str(metadata["team"])
                member = int(metadata.get("member", 0))
                turn = int(metadata.get("turn", 0))
                if not team or ":" in team or member < 0 or turn < 0:
                    raise ValueError
            except (TypeError, ValueError):
                self._error(400, "invalid teamcache metadata")
                return
            rid = f"{team}:{member}:{turn}"
            body["rid"] = rid
        explicit_team = team_from_rid(rid)
        if rid is not None and explicit_team is None:
            self._error(400, "rid must use team:member:turn with non-negative integers")
            return
        if explicit_team is not None:
            team_key = explicit_team
        elif body.get("messages"):
            row = describe_request(
                body, key=getattr(self.server, "fingerprint_key", None)
            )
            tenant = self.headers.get(self.server.tenant_header) or self.server.default_tenant
            try:
                ident = self.server.discovery.identify(
                    row, tenant, body.get("model") or "_", request_key
                )
                rid = ident["rid"]
                body["rid"] = rid
                team_key = ident["team_id"]
                completion_id = ident["completion_id"]
                discovery_status = "discovered"
            except Exception as exc:
                if self.server.discovery_required:
                    self._error(503, f"team discovery unavailable: {exc}")
                    return
                discovery_status = "bypass"
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        backend = self.server.router.choose(team_key or request_key)
        try:
            out_len, status = self._forward(backend, payload, rid, discovery_status)
        except Exception as exc:
            if completion_id is not None:
                try:
                    self.server.discovery.complete(completion_id, 0, success=False)
                except Exception:
                    pass
            self._error(502, f"upstream failed: {exc}")
            return
        if completion_id is not None:
            try:
                self.server.discovery.complete(
                    completion_id, out_len, success=status // 100 == 2
                )
            except Exception:
                pass  # availability first; discovery relearns after a missed completion

    def _proxy_without_discovery(self):
        backend = self.server.router.choose(self.path)
        try:
            self._forward(backend, b"", None, "bypass", method="GET")
        except Exception as exc:
            self._error(502, f"upstream failed: {exc}")

    def _forward(self, backend, payload, rid, discovery_status, method="POST"):
        parsed = urlsplit(backend)
        connection_cls = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_cls(
            parsed.hostname, parsed.port, timeout=self.server.upstream_timeout
        )
        base = parsed.path.rstrip("/")
        path = base + self.path
        headers = {
            key: value for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP
        }
        if method == "POST":
            headers["content-length"] = str(len(payload))
            headers["content-type"] = headers.get("content-type", "application/json")
        if rid:
            headers["x-request-id"] = rid
        connection.request(
            method,
            path,
            body=payload if method == "POST" else None,
            headers=headers,
        )
        response = connection.getresponse()
        content_type = response.getheader("content-type") or ""
        streaming = "text/event-stream" in content_type.lower()
        self.send_response(response.status, response.reason)
        for key, value in response.getheaders():
            if key.lower() in HOP_BY_HOP:
                continue
            self.send_header(key, value)
        self.send_header("x-teamcache-backend", backend)
        self.send_header("x-teamcache-discovery", discovery_status)
        if streaming:
            self.send_header("connection", "close")
            self.close_connection = True
            self.end_headers()
            counter = _SSECounter()
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                counter.feed(chunk)
                self.wfile.write(chunk)
                self.wfile.flush()
            connection.close()
            return counter.total, response.status
        data = response.read()
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        connection.close()
        try:
            return response_text_len(json.loads(data)), response.status
        except (TypeError, json.JSONDecodeError):
            return 0, response.status


def run_discovery_server(args) -> int:
    server = _Server((args.host, args.port), DiscoveryHandler)
    server.coordinator = DiscoveryCoordinator(
        TeamDiscovery(
            min_span_chars=args.min_span_chars,
            window_s=args.window_s,
            stuff_slack=args.stuff_slack,
        ),
        max_requests=args.max_requests,
        namespace=args.namespace,
    )
    server.token = os.environ.get("TEAMCACHE_DISCOVERY_TOKEN")
    server.max_body = args.max_body_mb * 1024 * 1024
    server.verbose = args.verbose
    print(f"TeamCache discovery listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


def run_gateway(args) -> int:
    token = os.environ.get("TEAMCACHE_DISCOVERY_TOKEN")
    if args.discovery_url:
        discovery = RemoteDiscoveryClient(
            args.discovery_url, token=token, timeout=args.discovery_timeout
        )
        discovery_desc = args.discovery_url
    else:
        discovery = LocalDiscoveryClient(DiscoveryCoordinator(
            TeamDiscovery(
                min_span_chars=args.min_span_chars,
                window_s=args.window_s,
                stuff_slack=args.stuff_slack,
            ),
            namespace=args.namespace,
        ))
        discovery_desc = "local"
    server = _Server((args.host, args.port), GatewayHandler)
    server.router = RendezvousRouter(args.backend)
    server.discovery = discovery
    server.discovery_required = args.discovery_required
    server.tenant_header = args.tenant_header.lower()
    server.default_tenant = args.default_tenant
    server.fingerprint_key = (
        os.environ.get("TEAMCACHE_FINGERPRINT_KEY") or token
    )
    server.upstream_timeout = args.upstream_timeout
    server.max_body = args.max_body_mb * 1024 * 1024
    server.verbose = args.verbose
    if args.health_interval > 0:
        thread = threading.Thread(
            target=_health_loop,
            args=(server.router, args.health_interval, args.health_timeout),
            daemon=True,
        )
        thread.start()
    print(
        f"TeamCache gateway listening on {args.host}:{args.port}; "
        f"backends={len(server.router.backends)} discovery={discovery_desc}",
        flush=True,
    )
    server.serve_forever()
    return 0


def _health_loop(router, interval, timeout):
    while True:
        for backend in router.backends:
            try:
                request = Request(backend.rstrip("/") + "/health", method="GET")
                with urlopen(request, timeout=timeout) as response:
                    healthy = response.status // 100 == 2
            except Exception:
                healthy = False
            router.mark_health(backend, healthy)
        time.sleep(interval)
