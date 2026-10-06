"""Command-line entry for installation, serving, and the bundled A/B case."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from teamcache import __version__
from teamcache.patch import (
    TARGET,
    PatchError,
    find_common_py,
    install,
    uninstall,
)
from teamcache.patch import (
    status as patch_status,
)
from teamcache.selftest import selftest


def _cmd_install(args) -> int:
    try:
        if args.out:
            src = Path(args.src) if args.src else find_common_py()
            dest = Path(args.out)
            if dest.is_dir() or str(dest).endswith(os.sep):
                dest = dest / "common.py"
            written = install(src=src, dest=dest)
            print(f"wrote {written}")
            print(f"bind-mount it over <sglang>/{TARGET}")
            print("set COHORT_POOL_TOKENS to your prefix-pool size in tokens")
            return 0
        written = install(src=Path(args.src) if args.src else None)
        print(f"patched {written}")
        print("restart the SGLang server, then set COHORT_POOL_TOKENS "
              "(or a control JSON) to your prefix-pool size in tokens")
        return 0
    except PatchError as exc:
        print(exc, file=sys.stderr)
        return 1


def _cmd_uninstall(args) -> int:
    try:
        path = uninstall(src=Path(args.src) if args.src else None)
        print(f"restored {path}")
        print("restart the SGLang server")
        return 0
    except PatchError as exc:
        print(exc, file=sys.stderr)
        return 1


def _cmd_status(args) -> int:
    info = patch_status(src=Path(args.src) if args.src else None)
    print(json.dumps(info, indent=2))
    if info.get("error"):
        return 1
    return 0


def _cmd_selftest(args) -> int:
    return selftest(tmpdir=args.tmpdir)


def _cmd_serve(args) -> int:
    from teamcache.serve import serve

    return serve(args)


def _cmd_benchmark(args) -> int:
    from teamcache.benchmark import run_ab

    return run_ab(args)


def _cmd_discovery(args) -> int:
    from teamcache.cluster import run_discovery_server

    return run_discovery_server(args)


def _cmd_gateway(args) -> int:
    from teamcache.cluster import run_gateway

    return run_gateway(args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="teamcache",
        description="Run TeamCache admission, discovery, and cluster routing for SGLang.",
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_install = sub.add_parser(
        "install",
        help="patch installed sglang/srt/mem_cache/common.py",
    )
    p_install.add_argument(
        "--src",
        help="read this common.py instead of the installed sglang copy",
    )
    p_install.add_argument(
        "--out",
        help="write patched common.py here (bind-mount) instead of in-place",
    )
    p_install.set_defaults(func=_cmd_install)

    p_un = sub.add_parser("uninstall", help="restore the pre-patch common.py")
    p_un.add_argument("--src", help="common.py path to restore")
    p_un.set_defaults(func=_cmd_uninstall)

    p_st = sub.add_parser("status", help="show whether SGLang is patched")
    p_st.add_argument("--src", help="common.py path to inspect")
    p_st.set_defaults(func=_cmd_status)

    p_test = sub.add_parser(
        "selftest",
        help="run policy checks without a GPU or SGLang server",
    )
    p_test.add_argument("--tmpdir", default=None)
    p_test.set_defaults(func=_cmd_selftest)

    p_serve = sub.add_parser(
        "serve",
        help="patch SGLang, configure TeamCache, and launch one model",
    )
    p_serve.add_argument("--model-path", required=True)
    p_serve.add_argument("--tp-size", type=int, default=1)
    p_serve.add_argument("--pool-tokens", type=int, required=True,
                         help="KV token budget; also passed to SGLang as max-total-tokens")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=30000)
    p_serve.add_argument("--chunked-prefill-size", type=int, default=65536)
    p_serve.add_argument(
        "--max-running-requests",
        type=int,
        default=None,
        help="optional in-flight request cap; useful when active KV crowds out the cache",
    )
    p_serve.add_argument("--control", default=None,
                         help="hot-reload JSON path (default: ~/.cache/teamcache/control.json)")
    p_serve.add_argument("--no-patch", action="store_true",
                         help="require an already-patched SGLang instead of patching it")
    p_serve.add_argument("--dry-run", action="store_true",
                         help="print the launch command without starting the server")
    p_serve.add_argument("sglang_args", nargs=argparse.REMAINDER,
                         help="extra SGLang arguments after --")
    p_serve.set_defaults(func=_cmd_serve)

    p_bench = sub.add_parser(
        "benchmark",
        help="run the bundled ring case against LRU and TeamCache",
    )
    p_bench.add_argument("--endpoint", default="http://127.0.0.1:30000")
    p_bench.add_argument("--dataset", default=None,
                         help="dataset JSON (default: bundled ring_20x5 case)")
    p_bench.add_argument("--control", default=None,
                         help="same control JSON used by teamcache serve")
    p_bench.add_argument("--output", default="teamcache-ab.json")
    p_bench.add_argument("--timeout", type=float, default=1800.0)
    p_bench.add_argument("--seed", type=int, default=None)
    p_bench.add_argument("--vocab-size", type=int, default=None)
    p_bench.add_argument("--pool-tokens", type=int, default=None,
                         help="override the dataset KV budget; must match the server")
    p_bench.add_argument(
        "--generation-tokens", type=int, default=None,
        help="override generated tokens per request; use 1 for input-only pressure",
    )
    p_bench.add_argument("--arm", choices=("both", "lru", "teamcache"), default="both",
                         help="run both arms (default) or one arm for capacity calibration")
    p_bench.add_argument(
        "--order", choices=("lru-first", "teamcache-first"), default="lru-first",
        help="execution order when --arm=both (default: lru-first)",
    )
    p_bench.add_argument("--quick", action="store_true",
                         help="2 teams, 2 rounds: connectivity check, not a performance result")
    p_bench.set_defaults(func=_cmd_benchmark)

    p_discovery = sub.add_parser(
        "discovery",
        help="run the privacy-safe cluster-wide team discovery coordinator",
    )
    p_discovery.add_argument("--host", default="127.0.0.1")
    p_discovery.add_argument("--port", type=int, default=39090)
    p_discovery.add_argument("--min-span-chars", type=int, default=1024)
    p_discovery.add_argument("--window-s", type=float, default=7200.0)
    p_discovery.add_argument("--stuff-slack", type=int, default=64)
    p_discovery.add_argument("--max-requests", type=int, default=100000)
    p_discovery.add_argument("--namespace", default=None,
                             help="request-id namespace; default is a random restart epoch")
    p_discovery.add_argument("--max-body-mb", type=int, default=8)
    p_discovery.add_argument("--verbose", action="store_true")
    p_discovery.set_defaults(func=_cmd_discovery)

    p_gateway = sub.add_parser(
        "gateway",
        help="discover teams, inject request ids, and route teams to a GPU cluster",
    )
    p_gateway.add_argument("--host", default="0.0.0.0")
    p_gateway.add_argument("--port", type=int, default=8000)
    p_gateway.add_argument("--backend", action="append", required=True,
                           help="SGLang base URL; repeat for every cluster node")
    p_gateway.add_argument("--discovery-url", default=None,
                           help="shared coordinator URL; omit for single-gateway local mode")
    p_gateway.add_argument("--discovery-required", action="store_true",
                           help="return 503 instead of bypassing when discovery is unavailable")
    p_gateway.add_argument("--discovery-timeout", type=float, default=5.0)
    p_gateway.add_argument("--upstream-timeout", type=float, default=1800.0)
    p_gateway.add_argument("--health-interval", type=float, default=10.0,
                           help="seconds between backend /health probes; 0 disables")
    p_gateway.add_argument("--health-timeout", type=float, default=2.0)
    p_gateway.add_argument("--tenant-header", default="x-teamcache-tenant")
    p_gateway.add_argument("--default-tenant", default="default")
    p_gateway.add_argument("--namespace", default=None,
                           help="local-discovery namespace; default is a random restart epoch")
    p_gateway.add_argument("--min-span-chars", type=int, default=1024)
    p_gateway.add_argument("--window-s", type=float, default=7200.0)
    p_gateway.add_argument("--stuff-slack", type=int, default=64)
    p_gateway.add_argument("--max-body-mb", type=int, default=32)
    p_gateway.add_argument("--verbose", action="store_true")
    p_gateway.set_defaults(func=_cmd_gateway)

    args = ap.parse_args(argv)
    code = args.func(args)
    if argv is None:
        raise SystemExit(code)
    return code


if __name__ == "__main__":
    main()
