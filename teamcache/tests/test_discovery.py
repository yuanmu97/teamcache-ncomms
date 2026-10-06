"""Team Discovery is optional and not wired into the engine patch."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from teamcache.discovery import self_test


class DiscoverySelftest(unittest.TestCase):
    def test_lineage_and_stuffed_handoff(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self_test()
        self.assertIn("self-test ok", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
