"""Policy self-test plus a synthetic common.py patch."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from teamcache.patch import (
    ANCHOR,
    PatchError,
    apply_text,
    backup_path,
    install,
    is_patched,
    status,
    uninstall,
)
from teamcache.policy import HELPER
from teamcache.selftest import selftest

FAKE_COMMON = f'''
{ANCHOR}

def cache_finished_req(req, tree_cache):
    tree_cache.cache_finished_req(
        req,
        is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
    )

def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    if getattr(req, "skip_radix_cache_insert", False):
        return

    tree_cache.cache_unfinished_req(req, **kwargs)
'''


class PolicySelftest(unittest.TestCase):
    def test_default_mode_is_unified(self):
        for key in ("COHORT_MODE", "COHORT_CONTROL", "COHORT_POOL_TOKENS"):
            os.environ.pop(key, None)
        ns = {}
        exec(compile(HELPER, "<cohort-helper>", "exec"), ns)
        self.assertEqual(ns["_cohort_cfg"].mode, "unified")

    def test_unified_and_control_reload(self):
        self.assertEqual(selftest(), 0)


class PatchSites(unittest.TestCase):
    def test_apply_text_is_idempotent_guard(self):
        out = apply_text(FAKE_COMMON)
        self.assertIn("_cohort_skip_insert", out)
        self.assertIn("not kwargs.get(\"chunked\", False)", out)
        with self.assertRaises(PatchError):
            apply_text(out)

    def test_install_backup_and_uninstall(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "common.py"
            src.write_text(FAKE_COMMON, encoding="utf-8")
            written = install(src=src)
            self.assertEqual(written, src)
            self.assertTrue(is_patched(src))
            bak = backup_path(src)
            self.assertTrue(bak.is_file())
            self.assertEqual(bak.read_text(encoding="utf-8"), FAKE_COMMON)
            info = status(src=src)
            self.assertTrue(info["patched"])
            uninstall(src=src)
            self.assertEqual(src.read_text(encoding="utf-8"), FAKE_COMMON)


if __name__ == "__main__":
    unittest.main()
