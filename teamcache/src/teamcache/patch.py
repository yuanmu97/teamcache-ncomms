"""Patch SGLang's radix insert sites with the TeamCache predicate.

Bypass has to be decided where the engine inserts into the prefix tree,
not at a gateway. This module rewrites `srt/mem_cache/common.py` at the
two insert sites, leaving the chunked-prefill accounting path alone.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from teamcache.policy import HELPER

TARGET = "srt/mem_cache/common.py"
ANCHOR = "from sglang.srt.utils.common import ceil_align"
BACKUP_SUFFIX = ".teamcache.bak"
MARKER = "_cohort_skip_insert"

EDITS = [
    ('is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),',
     'is_insert=is_insert and not _cohort_skip_insert(req),'),
    ('''def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    if getattr(req, "skip_radix_cache_insert", False):
        return

    tree_cache.cache_unfinished_req(req, **kwargs)''',
     '''def maybe_cache_unfinished_req(req: Req, tree_cache: BasePrefixCache, **kwargs):
    if getattr(req, "skip_radix_cache_insert", False):
        return

    # Decline retention here too, but never on the chunked path: that call is
    # load-bearing accounting, not a retention decision.
    if not kwargs.get("chunked", False) and _cohort_skip_insert(req):
        return

    tree_cache.cache_unfinished_req(req, **kwargs)'''),
]


class PatchError(Exception):
    """Raised when the installed SGLang source cannot be patched safely."""


def find_common_py() -> Path:
    try:
        import sglang
    except ImportError as exc:
        raise PatchError(
            "sglang is not installed; install it first, then retry"
        ) from exc
    root = Path(sglang.__file__).resolve().parent
    path = root / "srt" / "mem_cache" / "common.py"
    if not path.is_file():
        raise PatchError(f"expected {path} inside the sglang package")
    return path


def backup_path(src: Path) -> Path:
    return Path(str(src) + BACKUP_SUFFIX)


def apply_text(src: str) -> str:
    if MARKER in src:
        raise PatchError("source already contains TeamCache")
    if src.count(ANCHOR) != 1:
        raise PatchError(
            f"import anchor not found exactly once: {ANCHOR!r}. "
            "This SGLang version is not supported by this patch."
        )
    out = src
    for old, new in EDITS:
        if out.count(old) != 1:
            raise PatchError(
                f"edit site not found exactly once ({out.count(old)}). "
                "This SGLang version is not supported by this patch."
            )
        out = out.replace(old, new, 1)
    out = out.replace(ANCHOR, ANCHOR + HELPER, 1)
    for _, new in EDITS:
        if out.count(new) != 1:
            raise PatchError("rewritten site missing or duplicated")
    if 'getattr(req, "skip_radix_cache_insert", False)' not in out:
        raise PatchError("helper lost the original skip attribute check")
    compile(out, TARGET, "exec")
    return out


def is_patched(src: str | Path) -> bool:
    text = src.read_text(encoding="utf-8") if isinstance(src, Path) else src
    return MARKER in text


def install(src: Path | None = None, dest: Path | None = None) -> Path:
    """Patch installed (or given) common.py. Returns the written path."""
    path = Path(src) if src is not None else find_common_py()
    text = path.read_text(encoding="utf-8")
    patched = apply_text(text)
    out = Path(dest) if dest is not None else path
    if dest is None and not is_patched(text):
        shutil.copy2(path, backup_path(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(patched, encoding="utf-8")
    return out


def uninstall(src: Path | None = None) -> Path:
    path = Path(src) if src is not None else find_common_py()
    bak = backup_path(path)
    if not bak.is_file():
        raise PatchError(f"no backup at {bak}; cannot uninstall")
    shutil.copy2(bak, path)
    return path


def status(src: Path | None = None) -> dict:
    info = {
        "sglang": None,
        "common_py": None,
        "patched": False,
        "backup": None,
        "error": None,
    }
    try:
        import sglang
        info["sglang"] = str(Path(sglang.__file__).resolve())
    except ImportError:
        if src is None:
            info["error"] = "sglang is not installed"
            return info
    try:
        path = Path(src) if src is not None else find_common_py()
    except PatchError as exc:
        info["error"] = str(exc)
        return info
    info["common_py"] = str(path)
    info["patched"] = is_patched(path)
    bak = backup_path(path)
    info["backup"] = str(bak) if bak.is_file() else None
    info["env"] = {
        "COHORT_CONTROL": os.environ.get("COHORT_CONTROL"),
        "COHORT_POOL_TOKENS": os.environ.get("COHORT_POOL_TOKENS"),
        "COHORT_MODE": os.environ.get("COHORT_MODE"),
    }
    ctl = os.environ.get("COHORT_CONTROL")
    if ctl and Path(ctl).is_file():
        try:
            info["control"] = json.loads(Path(ctl).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            info["control"] = {"path": ctl, "error": str(exc)}
    else:
        info["control"] = None
    return info
