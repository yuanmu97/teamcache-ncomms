"""Public, procedural ring workload and isolated LRU-vs-TeamCache A/B runner."""

from __future__ import annotations

import json
import os
import random
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from pathlib import Path

from teamcache.serve import default_control_path


def bundled_dataset_path() -> Path:
    return Path(str(files("teamcache").joinpath("data/ring_20x5.json")))


def load_dataset(path: str | None) -> dict:
    target = Path(path).expanduser() if path else bundled_dataset_path()
    doc = json.loads(target.read_text(encoding="utf-8"))
    required = (
        "teams", "members", "rounds", "preamble_tokens", "message_tokens",
        "generation_tokens", "pool_tokens", "vocab_size", "seed",
    )
    missing = [key for key in required if key not in doc]
    if missing:
        raise ValueError(f"dataset misses required fields: {', '.join(missing)}")
    return doc


def _post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _flush(endpoint: str, timeout: float) -> None:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/flush_cache", data=b"", method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=min(timeout, 60.0)) as response:
            if response.status // 100 != 2:
                raise RuntimeError(f"flush_cache returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"SGLang /flush_cache failed: HTTP {exc.code}") from exc


def _write_control(path: Path, pool_tokens: int, enabled: bool) -> None:
    doc = {
        "mode": "unified" if enabled else "off",
        "quota": 0,
        "pool_tokens": pool_tokens,
        "window_s": 30,
        "adapt_window": 40,
        "adapt_margin": 1.02,
        "adapt_min_samples": 2,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _private_prompt(seed: int, team: int, member: int, length: int, vocab: int) -> list[int]:
    rng = random.Random((seed << 24) ^ (team << 12) ^ member)
    return [rng.randrange(256, vocab) for _ in range(length)]


def _transcript(seed: int, team: int, length: int, vocab: int) -> list[int]:
    rng = random.Random((seed << 24) ^ (team << 12) ^ 0xA51)
    return [rng.randrange(256, vocab) for _ in range(length)]


def generate_shape(dataset: dict) -> list[dict]:
    """Return public metadata rows; prompt tokens are generated only at run time."""
    rows = []
    for team in range(dataset["teams"]):
        for step in range(dataset["members"] * dataset["rounds"]):
            rows.append({
                "team": team,
                "member": step % dataset["members"],
                "turn": step // dataset["members"],
                "prompt_tokens": dataset["preamble_tokens"] + step * dataset["message_tokens"],
            })
    return rows


def _run_arm(endpoint: str, dataset: dict, timeout: float, label: str) -> dict:
    generate_url = endpoint.rstrip("/") + "/generate"
    teams = dataset["teams"]
    members = dataset["members"]
    rounds = dataset["rounds"]
    preamble = dataset["preamble_tokens"]
    message = dataset["message_tokens"]
    vocab = dataset["vocab_size"]
    seed = dataset["seed"]
    total_steps = members * rounds
    transcript_len = total_steps * message

    private = {
        (t, m): _private_prompt(seed, t, m, preamble, vocab)
        for t in range(teams) for m in range(members)
    }
    transcripts = {
        t: _transcript(seed, t, transcript_len, vocab) for t in range(teams)
    }

    def run_team(team: int) -> list[dict]:
        rows = []
        for step in range(total_steps):
            member = step % members
            turn = step // members
            prompt = private[(team, member)] + transcripts[team][: step * message]
            started = time.monotonic()
            body = _post(generate_url, {
                "input_ids": prompt,
                "sampling_params": {
                    "max_new_tokens": dataset["generation_tokens"],
                    "temperature": 0,
                },
                "rid": f"team{team}:{member}:{turn}",
            }, timeout)
            latency = time.monotonic() - started
            meta = body.get("meta_info") or {}
            prompt_tokens = int(meta.get("prompt_tokens") or len(prompt))
            cached_tokens = int(meta.get("cached_tokens") or 0)
            rows.append({
                "team": team,
                "member": member,
                "turn": turn,
                "step": step,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "latency_s": latency,
            })
        return rows

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=teams) as pool:
        nested = list(pool.map(run_team, range(teams)))
    wall = time.monotonic() - started
    rows = [row for team_rows in nested for row in team_rows]
    steady_from = dataset.get("steady_from_round", max(rounds // 2, 1))
    steady = [row for row in rows if row["turn"] >= steady_from]

    def summarize(sample: list[dict]) -> dict:
        prompts = sum(row["prompt_tokens"] for row in sample)
        cached = sum(row["cached_tokens"] for row in sample)
        return {
            "requests": len(sample),
            "prompt_tokens": prompts,
            "cached_tokens": cached,
            "prefix_hit_rate": cached / prompts if prompts else 0.0,
            "request_latency_p50_s": statistics.median(row["latency_s"] for row in sample),
        }

    return {
        "label": label,
        "wall_s": wall,
        "full": summarize(rows),
        "steady": summarize(steady),
        "steady_from_round": steady_from,
    }


def _activate(
    control: Path, endpoint: str, pool_tokens: int, enabled: bool, timeout: float
) -> None:
    _write_control(control, pool_tokens, enabled)
    # refresh() is rate-limited to one stat per second. Wait past that window,
    # then make an unlabelled request that applies the file without polluting
    # the TeamCache team state. Flush after it so each arm starts cold.
    time.sleep(1.1)
    _post(endpoint.rstrip("/") + "/generate", {
        "input_ids": [271, 828, 314],
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
    }, min(timeout, 60.0))
    _flush(endpoint, timeout)


def arm_sequence(arm: str, order: str) -> tuple[str, ...]:
    """Return the requested arms in an explicit, auditable execution order."""
    if arm != "both":
        return (arm,)
    if order == "teamcache-first":
        return ("teamcache", "lru")
    return ("lru", "teamcache")


def run_ab(args) -> int:
    dataset = load_dataset(args.dataset)
    if args.seed is not None:
        dataset["seed"] = args.seed
    if args.vocab_size is not None:
        dataset["vocab_size"] = args.vocab_size
    if args.pool_tokens is not None:
        if args.pool_tokens < 1:
            raise ValueError("pool_tokens must be positive")
        dataset["pool_tokens"] = args.pool_tokens
    if args.generation_tokens is not None:
        if args.generation_tokens < 1:
            raise ValueError("generation_tokens must be positive")
        dataset["generation_tokens"] = args.generation_tokens
    if dataset["vocab_size"] <= 256:
        raise ValueError("vocab_size must be greater than 256")
    if args.quick:
        dataset.update({
            "teams": 2,
            "rounds": 2,
            "generation_tokens": 1,
            "steady_from_round": 1,
        })
    control = Path(args.control).expanduser() if args.control else default_control_path()

    print(f"dataset={dataset['name']} teams={dataset['teams']} members={dataset['members']} "
          f"rounds={dataset['rounds']} pool={dataset['pool_tokens']:,} tokens")
    order = getattr(args, "order", "lru-first")
    arms = {}
    for arm in arm_sequence(args.arm, order):
        enabled = arm == "teamcache"
        print("arm: TeamCache" if enabled else "arm: upstream LRU")
        _activate(control, args.endpoint, dataset["pool_tokens"], enabled, args.timeout)
        arms[arm] = _run_arm(args.endpoint, dataset, args.timeout, arm)

    comparison = None
    if args.arm == "both":
        lru_hit = arms["lru"]["steady"]["prefix_hit_rate"]
        tc_hit = arms["teamcache"]["steady"]["prefix_hit_rate"]
        comparison = {
            "steady_hit_rate_gain_pp": (tc_hit - lru_hit) * 100,
            "wall_speedup": arms["lru"]["wall_s"] / arms["teamcache"]["wall_s"]
            if arms["teamcache"]["wall_s"] else None,
        }
    result = {
        "schema_version": 1,
        "dataset": dataset,
        "endpoint": args.endpoint,
        "arm_order": list(arm_sequence(args.arm, order)),
        "arms": arms,
        "comparison": comparison,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if "lru" in arms:
        print(f"LRU       steady hit {arms['lru']['steady']['prefix_hit_rate'] * 100:6.2f}%  "
              f"wall {arms['lru']['wall_s']:.1f}s")
    if "teamcache" in arms:
        print(f"TeamCache steady hit {arms['teamcache']['steady']['prefix_hit_rate'] * 100:6.2f}%  "
              f"wall {arms['teamcache']['wall_s']:.1f}s")
    if comparison is not None:
        print(f"gain {comparison['steady_hit_rate_gain_pp']:+.2f} pp, "
              f"speedup {comparison['wall_speedup']:.2f}x")
    print(f"wrote {output}")
    return 0
