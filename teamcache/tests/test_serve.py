from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from teamcache.serve import build_command, write_control


class ServeCommand(unittest.TestCase):
    def args(self, **overrides):
        values = {
            "model_path": "/models/demo",
            "tp_size": 8,
            "pool_tokens": 420800,
            "host": "0.0.0.0",
            "port": 30000,
            "chunked_prefill_size": 65536,
            "max_running_requests": None,
            "sglang_args": ["--", "--trust-remote-code"],
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_build_command_shares_pool_budget_with_engine(self):
        command = build_command(self.args())
        self.assertIn("sglang.launch_server", command)
        self.assertEqual(command[command.index("--max-total-tokens") + 1], "420800")
        self.assertEqual(command[-1], "--trust-remote-code")

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            build_command(self.args(pool_tokens=0))

    def test_optional_running_request_cap_is_forwarded(self):
        command = build_command(self.args(max_running_requests=20))
        self.assertEqual(command[command.index("--max-running-requests") + 1], "20")

    def test_invalid_running_request_cap_is_rejected(self):
        with self.assertRaises(ValueError):
            build_command(self.args(max_running_requests=0))

    def test_control_is_atomic_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "control.json"
            write_control(path, 12345)
            doc = json.loads(path.read_text())
            self.assertEqual(doc["mode"], "unified")
            self.assertEqual(doc["pool_tokens"], 12345)
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
