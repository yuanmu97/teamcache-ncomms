#!/usr/bin/env python3
"""Start a long-lived helper fully detached from the calling shell.

The tunnel and the recording proxy have to outlive the shell that starts
them, and on macOS `nohup ... &` from a tool-driven shell is not enough --
the process still dies with its group. Double-fork into a new session so it
does not.

  python3 spawn.py --log /tmp/x.log -- python3 proxy_openai.py --trace ...
"""

from __future__ import annotations

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    cfg = ap.parse_args()
    cmd = cfg.cmd[1:] if cfg.cmd and cfg.cmd[0] == "--" else cfg.cmd
    if not cmd:
        raise SystemExit("no command")

    if os.fork() > 0:
        os.wait()
        return
    os.setsid()
    if os.fork() > 0:
        os._exit(0)

    # A detached process must not keep its caller's descriptors. The callers
    # here hold `flock`s for mutual exclusion -- `e2e_arms.sh` on fd 9,
    # `e2e_autogen.sh` on fd 8 -- and a recording proxy that inherits one
    # outlives the script that took it, so the lock is never released. That
    # cost a run: the first arm succeeded and the other two were refused by a
    # lock held by the first arm's own proxy, which reads as "another instance
    # is running" when nothing is.
    os.closerange(3, 4096)

    fd = os.open(cfg.log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(os.open(os.devnull, os.O_RDONLY))
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    print(f"pid {os.getpid()}: {' '.join(cmd)}", flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
