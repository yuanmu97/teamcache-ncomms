"""One-command SGLang launcher.

The launcher intentionally exposes only the settings TeamCache must know. Any
model-specific SGLang flags can be appended after ``--``.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

from teamcache.patch import PatchError, find_common_py, install, is_patched


def default_control_path() -> Path:
    return Path.home() / ".cache" / "teamcache" / "control.json"


def write_control(path: Path, pool_tokens: int) -> None:
    doc = {
        "mode": "unified",
        "pool_tokens": int(pool_tokens),
        "window_s": 30,
        "adapt_window": 40,
        "adapt_margin": 1.02,
        "adapt_min_samples": 2,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def build_command(args) -> list[str]:
    if args.pool_tokens <= 0:
        raise ValueError("--pool-tokens must be greater than zero")
    if args.tp_size <= 0:
        raise ValueError("--tp-size must be greater than zero")
    max_running = getattr(args, "max_running_requests", None)
    if max_running is not None and max_running <= 0:
        raise ValueError("--max-running-requests must be greater than zero")
    extra = list(args.sglang_args or [])
    if extra[:1] == ["--"]:
        extra = extra[1:]
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--tp-size",
        str(args.tp_size),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-total-tokens",
        str(args.pool_tokens),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--enable-cache-report",
    ]
    if max_running is not None:
        command.extend(["--max-running-requests", str(max_running)])
    command.extend(extra)
    return command


def serve(args) -> int:
    try:
        command = build_command(args)
    except ValueError as exc:
        print(f"teamcache: {exc}", file=sys.stderr)
        return 2

    try:
        common = find_common_py()
        if not is_patched(common):
            if args.no_patch:
                raise PatchError(
                    f"{common} is not patched; run `teamcache install` or remove --no-patch"
                )
            install(src=common)
            print(f"patched {common} (backup: {common}.teamcache.bak)")
    except PatchError as exc:
        print(f"teamcache: {exc}", file=sys.stderr)
        return 2

    control = Path(args.control).expanduser() if args.control else default_control_path()
    write_control(control, args.pool_tokens)
    print(f"control: {control}")
    print("launch:", shlex.join(command))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    # COHORT_* is the stable internal name used by the evaluated engine hook.
    # The public CLI keeps it out of the user-facing configuration surface.
    env["COHORT_CONTROL"] = str(control)
    env["COHORT_POOL_TOKENS"] = str(args.pool_tokens)
    os.execvpe(command[0], command, env)
    return 0  # pragma: no cover - exec only returns on failure
