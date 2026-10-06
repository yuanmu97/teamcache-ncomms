#!/usr/bin/env python3
"""Same admission rule, hooked at vLLM's prefix-cache insert.

vLLM already has a selective-cache hook: `BlockPool.cache_full_blocks` takes
`request` and an optional `block_mask`. Do NOT early-return: the same
request_id is applied a dozen times (DCP / kv group / chunk), and skipping
the function freezes later hash inserts. Drop means `block_mask` all-False
so the block never enters `cached_block_hash_to_block` and is prepended on
free (evicted first). Keep/pass leave the mask alone.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

# Reuse the SGLang helper text, then rewrite the three engine-specific sites.
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import patch_sglang_cohort as S  # noqa: E402

TARGET = "vllm/v1/core/block_pool.py"
ANCHOR = "class BlockPool:"

HELPER_FIXES = (
    ('rid = getattr(req, "rid", None) or ""',
     'rid = (getattr(req, "request_id", None) or getattr(req, "rid", None) or "")'),
    ('for _name in ("origin_input_ids", "origin_input_ids_unpadded",\n'
     '                      "full_untruncated_fill_ids"):\n'
     '            _ids = getattr(req, _name, None)\n'
     '            if _ids is not None:\n'
     '                try:\n'
     '                    tokens = max(tokens, len(_ids))\n'
     '                except TypeError:\n'
     '                    pass',
     'for _name in ("num_tokens", "num_prompt_tokens"):\n'
     '            _n = getattr(req, _name, None)\n'
     '            if isinstance(_n, int) and _n > 0:\n'
     '                tokens = max(tokens, _n)\n'
     '        for _name in ("prompt_token_ids", "all_token_ids",\n'
     '                      "origin_input_ids"):\n'
     '            _ids = getattr(req, _name, None)\n'
     '            if _ids is not None:\n'
     '                try:\n'
     '                    tokens = max(tokens, len(_ids))\n'
     '                except TypeError:\n'
     '                    pass'),
    ('ids = getattr(req, "origin_input_ids", None)\n'
     '        st.size[(team, member)] = len(ids) if ids else 0',
     'ids = getattr(req, "prompt_token_ids", None) or getattr(req, "origin_input_ids", None)\n'
     '        ntok = getattr(req, "num_tokens", 0) or 0\n'
     '        try:\n'
     '            ntok = max(int(ntok), len(ids) if ids is not None else 0)\n'
     '        except TypeError:\n'
     '            pass\n'
     '        st.size[(team, member)] = ntok'),
)

SITE = '''    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:'''

SITE_NEW = '''    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        # vLLM-native drop: mask out hash insert, then fall through so
        # refcount / partial-block bookkeeping still run. once[rid] only
        # caches this decision; never return here.
        if _cohort_skip_insert(request):
            _n = num_full_blocks - num_cached_blocks
            if _n > 0:
                _extra = [False] * _n
                if block_mask is None:
                    block_mask = _extra
                else:
                    block_mask = [a and b for a, b in zip(block_mask, _extra)]'''


def helper_for_vllm() -> str:
    text = S.HELPER
    for old, new in HELPER_FIXES:
        if old not in text:
            raise SystemExit(f"helper site missing: {old[:60]!r}")
        text = text.replace(old, new, 1)
    return text


def patch(src: str) -> str:
    if "_cohort_skip_insert" in src:
        raise SystemExit("source already patched")
    if src.count(ANCHOR) != 1:
        raise SystemExit("BlockPool class not found exactly once")
    if src.count(SITE) != 1:
        raise SystemExit("cache_full_blocks signature not found exactly once")
    helper = helper_for_vllm()
    out = src.replace(SITE, SITE_NEW, 1)
    out = out.replace(ANCHOR, helper + "\n\n" + ANCHOR, 1)
    if out.count("_cohort_skip_insert") < 2:
        raise SystemExit("helper or call site missing after patch")
    if "if _cohort_skip_insert(request):\n            return" in out:
        raise SystemExit("early-return hook leaked back in")
    if "block_mask = _extra" not in out:
        raise SystemExit("block_mask drop path missing after patch")
    return out


def read_from_image(image: str) -> str:
    script = (
        'P=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))"); '
        f'cat $P/v1/core/block_pool.py'
    )
    out = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "bash", image, "-c", script],
        capture_output=True, text=True, check=True)
    return out.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="vllm-kimi-k3:5894fdf")
    ap.add_argument("--src")
    ap.add_argument("--out", default="/tmp/vllm-block_pool-cohort.py")
    ap.add_argument("--check", action="store_true")
    cfg = ap.parse_args()
    src = pathlib.Path(cfg.src).read_text() if cfg.src else read_from_image(cfg.image)
    out = patch(src)
    pathlib.Path(cfg.out).write_text(out)
    print(f"wrote {cfg.out} ({len(out)} bytes)")
    if cfg.check:
        ns = {}
        # The helper references Request only at the call site; compile the
        # helper body by extracting it.
        start = out.find("import json as _cohort_json")
        end = out.find("class BlockPool")
        exec(compile(out[start:end], "<vllm-helper>", "exec"), ns)
        print("helper compiled", sorted(k for k in ns if k.startswith("_cohort") or k.startswith("_Cohort")))


if __name__ == "__main__":
    main()
