from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler
from urllib.request import Request, urlopen

from teamcache.cluster import (
    DiscoveryCoordinator,
    DiscoveryHandler,
    GatewayHandler,
    RemoteDiscoveryClient,
    RendezvousRouter,
    _Server,
    describe_request,
    team_from_rid,
)
from teamcache.discovery import TeamDiscovery


class _FakeBackendHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.seen.append(body.get("rid"))
        text = "x" * 120
        data = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": text}}]
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _start(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


class ClusterDiscoveryTests(unittest.TestCase):
    def test_describe_request_never_returns_prompt_text(self):
        body = {"model": "m", "messages": [
            {"role": "system", "content": "private system"},
            {"role": "user", "content": "private user"},
        ]}
        row = describe_request(body, now=1)
        encoded = json.dumps(row)
        self.assertNotIn("private system", encoded)
        self.assertNotIn("private user", encoded)
        self.assertEqual(row["msg_sizes"], [14, 12])
        self.assertEqual(len(row["prefix_sha"]), 2)

    def test_fingerprint_key_prevents_offline_plain_hash_matching(self):
        body = {"messages": [{"role": "user", "content": "common prompt"}]}
        plain = describe_request(body, now=1)
        keyed = describe_request(body, now=1, key="cluster secret")
        same_key = describe_request(body, now=1, key="cluster secret")
        self.assertNotEqual(plain["prefix_sha"], keyed["prefix_sha"])
        self.assertEqual(keyed["prefix_sha"], same_key["prefix_sha"])

    def test_coordinator_is_two_phase_and_idempotent(self):
        coordinator = DiscoveryCoordinator(TeamDiscovery(min_span_chars=50))
        a = {"t_recv": 1, "prefix_sha": ["a", "au"],
             "msg_sizes": [80, 40], "msg_roles": ["system", "user"]}
        first = coordinator.identify(a, "tenant", "model", "request-a")
        retry = coordinator.identify(a, "tenant", "model", "request-a")
        self.assertEqual(first, retry)
        self.assertTrue(coordinator.complete(first["completion_id"], 120))
        replay = coordinator.identify(a, "tenant", "model", "request-a")
        self.assertEqual(first["rid"], replay["rid"])
        self.assertNotEqual(first["completion_id"], replay["completion_id"])
        self.assertTrue(coordinator.complete(replay["completion_id"], 120))

        b = {"t_recv": 2, "prefix_sha": ["b", "bu", "bua"],
             "msg_sizes": [80, 40, 120],
             "msg_roles": ["system", "user", "user"]}
        second = coordinator.identify(b, "tenant", "model", "request-b")
        self.assertEqual(first["team_id"], second["team_id"])
        self.assertNotEqual(first["member_id"], second["member_id"])
        self.assertEqual(coordinator.stats()["handoffs"], 1)

        a2 = {"t_recv": 3, "prefix_sha": ["a", "au", "aua"],
              "msg_sizes": [80, 40, 120],
              "msg_roles": ["system", "user", "user"]}
        third = coordinator.identify(a2, "tenant", "model", "request-a2")
        self.assertEqual(first["member_id"], third["member_id"])
        self.assertEqual(third["turn"], 1)

    def test_indexed_extension_does_not_scan_lineages(self):
        discovery = TeamDiscovery(min_span_chars=10, window_s=1e9)
        for i in range(2000):
            discovery.observe({
                "t_recv": i / 1000,
                "prefix_sha": [f"s{i}", f"u{i}"],
                "msg_sizes": [20, 20],
                "msg_roles": ["system", "user"],
                "out_text_len": 20,
            })
        before = discovery.stats()["extension_probes"]
        identity = discovery.observe({
            "t_recv": 3,
            "prefix_sha": ["s0", "u0", "next"],
            "msg_sizes": [20, 20, 20],
            "msg_roles": ["system", "user", "user"],
            "out_text_len": 20,
        })
        self.assertEqual(identity.member_id, 0)
        self.assertLessEqual(discovery.stats()["extension_probes"] - before, 2)

    def test_rendezvous_is_stable_and_order_independent(self):
        a = RendezvousRouter(["http://n2:30000", "http://n1:30000"])
        b = RendezvousRouter(["http://n1:30000", "http://n2:30000"])
        self.assertEqual(a.choose("team-7"), b.choose("team-7"))
        self.assertEqual(a.choose("team-7"), a.choose("team-7"))
        a.mark_health("http://n1:30000", False)
        self.assertEqual(a.choose("team-7"), "http://n2:30000")

    def test_request_id_shape_is_strict(self):
        self.assertEqual(team_from_rid("alpha:2:7"), "alpha")
        self.assertIsNone(team_from_rid("alpha:member:7"))
        self.assertIsNone(team_from_rid("alpha:2:7:extra"))

    def test_gateway_rejects_malformed_explicit_request_id(self):
        try:
            backend = _Server(("127.0.0.1", 0), _FakeBackendHandler)
        except PermissionError:
            self.skipTest("loopback listeners are disabled in this sandbox")
        backend.seen = []
        gateway = None
        try:
            _start(backend)
            gateway = _Server(("127.0.0.1", 0), GatewayHandler)
            gateway.router = RendezvousRouter([
                f"http://127.0.0.1:{backend.server_port}"
            ])
            gateway.discovery = None
            gateway.discovery_required = True
            gateway.tenant_header = "x-teamcache-tenant"
            gateway.default_tenant = "default"
            gateway.upstream_timeout = 5
            gateway.max_body = 1024 * 1024
            gateway.verbose = False
            _start(gateway)
            data = json.dumps({
                "model": "m",
                "messages": [{"role": "user", "content": "hello"}],
                "rid": "not-a-team-id",
            }).encode()
            req = Request(
                f"http://127.0.0.1:{gateway.server_port}/v1/chat/completions",
                data=data,
                headers={"content-type": "application/json"},
                method="POST",
            )
            with self.assertRaisesRegex(Exception, "HTTP Error 400"):
                urlopen(req, timeout=5)
            self.assertEqual(backend.seen, [])
        finally:
            if gateway is not None:
                gateway.shutdown()
                gateway.server_close()
            backend.shutdown()
            backend.server_close()

    def test_coordinator_restart_epoch_prevents_id_collision(self):
        row = {"t_recv": 1, "prefix_sha": ["a"],
               "msg_sizes": [80], "msg_roles": ["system"]}
        one = DiscoveryCoordinator(namespace="epoch-one").identify(row, request_key="r")
        two = DiscoveryCoordinator(namespace="epoch-two").identify(row, request_key="r")
        self.assertNotEqual(one["team_id"], two["team_id"])

    def test_coordinator_ignores_gateway_clock_skew(self):
        coordinator = DiscoveryCoordinator(
            TeamDiscovery(min_span_chars=10, window_s=60)
        )
        first = coordinator.identify({
            "t_recv": 10**12,
            "prefix_sha": ["a", "au"],
            "msg_sizes": [20, 20],
            "msg_roles": ["system", "user"],
        }, request_key="clock-a")
        second = coordinator.identify({
            "t_recv": -10**12,
            "prefix_sha": ["a", "au", "next"],
            "msg_sizes": [20, 20, 20],
            "msg_roles": ["system", "user", "user"],
        }, request_key="clock-b")
        self.assertEqual(first["member_id"], second["member_id"])

    def test_idempotency_key_is_scoped_by_tenant_and_model(self):
        coordinator = DiscoveryCoordinator(TeamDiscovery(min_span_chars=10))
        row = {
            "prefix_sha": ["a", "au"],
            "msg_sizes": [20, 20],
            "msg_roles": ["system", "user"],
        }
        tenant_a = coordinator.identify(
            row, tenant="tenant-a", model="model-a", request_key="same-id"
        )
        tenant_b = coordinator.identify(
            row, tenant="tenant-b", model="model-a", request_key="same-id"
        )
        model_b = coordinator.identify(
            row, tenant="tenant-a", model="model-b", request_key="same-id"
        )
        self.assertNotEqual(tenant_a["member_id"], tenant_b["member_id"])
        self.assertNotEqual(tenant_a["member_id"], model_b["member_id"])

    def test_two_gateways_share_discovery_and_team_affinity(self):
        servers = []
        try:
            backends = []
            for _ in range(2):
                try:
                    backend = _Server(("127.0.0.1", 0), _FakeBackendHandler)
                except PermissionError:
                    self.skipTest("loopback listeners are disabled in this sandbox")
                backend.seen = []
                _start(backend)
                servers.append(backend)
                backends.append(f"http://127.0.0.1:{backend.server_port}")

            discovery = _Server(("127.0.0.1", 0), DiscoveryHandler)
            discovery.coordinator = DiscoveryCoordinator(
                TeamDiscovery(min_span_chars=50)
            )
            discovery.token = None
            discovery.max_body = 1024 * 1024
            discovery.verbose = False
            _start(discovery)
            servers.append(discovery)
            discovery_url = f"http://127.0.0.1:{discovery.server_port}"

            gateways = []
            for _ in range(2):
                gateway = _Server(("127.0.0.1", 0), GatewayHandler)
                gateway.router = RendezvousRouter(backends)
                gateway.discovery = RemoteDiscoveryClient(discovery_url)
                gateway.discovery_required = True
                gateway.tenant_header = "x-teamcache-tenant"
                gateway.default_tenant = "default"
                gateway.upstream_timeout = 5
                gateway.max_body = 1024 * 1024
                gateway.verbose = False
                _start(gateway)
                servers.append(gateway)
                gateways.append(gateway)

            def request(gateway, messages, request_id):
                data = json.dumps({"model": "m", "messages": messages}).encode()
                req = Request(
                    f"http://127.0.0.1:{gateway.server_port}/v1/chat/completions",
                    data=data,
                    headers={
                        "content-type": "application/json",
                        "x-request-id": request_id,
                        "x-teamcache-tenant": "tenant-a",
                    },
                    method="POST",
                )
                with urlopen(req, timeout=5) as response:
                    return response.headers["x-teamcache-backend"], json.load(response)

            first_backend, _ = request(gateways[0], [
                {"role": "system", "content": "A" * 80},
                {"role": "user", "content": "u" * 40},
            ], "a0")
            second_backend, _ = request(gateways[1], [
                {"role": "system", "content": "B" * 80},
                {"role": "user", "content": "v" * 40},
                {"role": "user", "content": "x" * 120},
            ], "b0")
            self.assertEqual(first_backend, second_backend)
            all_rids = [rid for backend in servers[:2] for rid in backend.seen]
            self.assertEqual(len(all_rids), 2)
            self.assertEqual(all_rids[0].split(":")[0], all_rids[1].split(":")[0])
        finally:
            for server in reversed(servers):
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
