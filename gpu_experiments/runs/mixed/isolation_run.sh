#!/usr/bin/env bash
# Isolation 4-arm mixed seed 11. Does not overwrite original gpu/mixed_*.json.
# Order: TeamCache → LRU → Static-rate → Group-blind.
# Unique change: physical /flush_cache + helper restore; no filler; extra dump fields.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$ROOT/../replay_team.py" ]]; then
  PAT="$(cd "$ROOT/.." && pwd)"
else
  PAT="${PAT:-/opt/dlami/nvme/pat}"
fi
CTRL="${CTRL:-/tmp/cohort/control.json}"
CONTAINER="${CONTAINER:-kimi-k3-b300-sglang-cohort}"
POOL=420800
MEMBERS=5
MIX_CAP=28
MIX_GCAP=24
MIX_Q=4
SEED=11
ENGINE=http://127.0.0.1:30000
OUT="$ROOT/gpu/isolation_s11"
RUNS="$OUT/isolation_runs.csv"
GPU_PER_NODE=8
# Remaining from prior 16 GPU-h budget: 16-10.8644 = 5.1356 GPU-h = 2311 node-s.
REMAIN_NODE_S="${REMAIN_NODE_S:-2311}"
ESTIMATE_NODE_S="${ESTIMATE_NODE_S:-1500}"  # 25 min conservative for 4 arms

mkdir -p "$OUT"
if [[ -f "$OUT/DONE" ]]; then
  echo "isolation_s11 already DONE; refusing to overwrite" >&2
  exit 3
fi

if (( ESTIMATE_NODE_S + 180 > REMAIN_NODE_S )); then
  echo "budget gate: estimate ${ESTIMATE_NODE_S}s + 180s margin > remain ${REMAIN_NODE_S}s; not starting incomplete group" >&2
  exit 4
fi

NODE_START=$(date +%s)
echo "ts,name,status,run_s,node_elapsed_s,gpu_hours,note" >"$RUNS"

log_run() {
  local now elapsed
  now=$(date +%s)
  elapsed=$(( now - NODE_START ))
  python3 -c "print('${now}')" >/dev/null
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$1,$2,$3,$elapsed,$(python3 -c "print(f'{$elapsed*$GPU_PER_NODE/3600:.4f}')"),${4:-}" >>"$RUNS"
}

budget_ok() {
  local now elapsed
  now=$(date +%s)
  elapsed=$(( now - NODE_START ))
  if (( elapsed > REMAIN_NODE_S - 60 )); then
    echo "budget exhausted elapsed=$elapsed remain=$REMAIN_NODE_S" >&2
    return 1
  fi
}

flush_cache() {
  local code
  code=$(curl -sS -o /tmp/isolation_flush.body -w "%{http_code}" -m 30 -X POST \
    "$ENGINE/flush_cache" || echo 000)
  echo "flush_cache http=$code body=$(head -c 240 /tmp/isolation_flush.body)"
  if [[ "$code" != "200" ]]; then
    echo "flush_cache failed" >&2
    return 1
  fi
  if ! grep -q "Cache flushed" /tmp/isolation_flush.body; then
    echo "flush_cache did not report success: $(cat /tmp/isolation_flush.body)" >&2
    return 1
  fi
  docker logs --since 15s "$CONTAINER" 2>&1 | grep -i "Cache flushed" | tail -3 || true
}

set_policy() {
  python3 - "$CTRL" "$1" "$2" "$3" "$4" "$5" "$POOL" <<'PY'
import json, sys, os
path, mode, quota, gcap, members, cap, pool = sys.argv[1:8]
os.makedirs(os.path.dirname(path), exist_ok=True)
doc = {
  "mode": mode, "quota": int(quota), "global_cap": int(gcap),
  "members": int(members), "capacity": int(cap),
  "rotate_period": 0, "window_s": 30, "pool_tokens": int(pool),
  "adapt_threshold": 0.6, "adapt_window": 40,
  "adapt_margin": 1.02, "adapt_min_samples": 2,
}
with open(path, "w") as fh:
    json.dump(doc, fh)
    fh.flush(); os.fsync(fh.fileno())
print("wrote", path, doc)
PY
  sleep 2
}

confirm_gen() {
  local want=$1
  local n
  sleep 1
  n=$(docker logs --since 20s "$CONTAINER" 2>&1 | grep -c "\[cohort\] gen=${want} " || true)
  echo "gen=$want rank_lines=$n (since 20s)"
  if [[ "$n" -lt 8 ]]; then
    echo "retry full-log count" >&2
    n=$(docker logs "$CONTAINER" 2>&1 | grep -c "\[cohort\] gen=${want} " || true)
    echo "gen=$want rank_lines=$n (full)"
  fi
  if [[ "$n" -lt 8 ]]; then
    docker logs --since 30s "$CONTAINER" 2>&1 | grep '\[cohort\] gen=' | tail -16 || true
    echo "expected 8 TP rank gen lines" >&2
    return 1
  fi
}

probe_empty() {
  local tag=$1
  python3 - "$ENGINE" "$OUT/probe_${tag}.json" <<'PY'
import json, random, sys, urllib.request, time
url = sys.argv[1] + "/generate"
outp = sys.argv[2]
rng = random.Random(time.time_ns())
ids = [rng.randrange(16, 100000) for _ in range(256)]
body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "rid": "isolation-probe"}
req = urllib.request.Request(url, data=json.dumps(body).encode(),
                             headers={"content-type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as resp:
    doc = json.loads(resp.read())
meta = doc.get("meta_info") or {}
rec = {"cached_tokens": meta.get("cached_tokens"),
       "prompt_tokens": meta.get("prompt_tokens"),
       "completion_tokens": meta.get("completion_tokens"),
       "meta_keys": sorted(meta.keys())}
open(outp, "w").write(json.dumps(rec, indent=2))
print("probe", rec)
cached = meta.get("cached_tokens") or 0
if cached != 0:
    raise SystemExit(f"probe cached_tokens={cached} after flush; isolation invalid")
PY
}

next_gen() {
  docker logs "$CONTAINER" 2>&1 | grep -o '\[cohort\] gen=[0-9]*' | sed 's/.*gen=//' | sort -n | tail -1
}

health() {
  curl -sf -m 5 "$ENGINE/health" >/dev/null
}

ping_refresh() {
  python3 - "$ENGINE" <<'PY'
import json, random, sys, time, urllib.request
url = sys.argv[1] + "/generate"
rng = random.Random(time.time_ns())
ids = [rng.randrange(16, 100000) for _ in range(64)]
body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "rid": "isolation-ping"}
req = urllib.request.Request(url, data=json.dumps(body).encode(),
                             headers={"content-type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as resp:
    doc = json.loads(resp.read())
meta = doc.get("meta_info") or {}
print("ping cached", meta.get("cached_tokens"), "completion", meta.get("completion_tokens"))
PY
}

confirm_mode() {
  local want_mode=$1
  local line
  sleep 1
  line=$(docker logs --since 30s "$CONTAINER" 2>&1 | grep "\[cohort\] gen=" | tail -1 || true)
  echo "latest gen line: ${line:-<none>}"
  if [[ "$line" != *"mode='$want_mode'"* && "$line" != *"mode=\"$want_mode\""* ]]; then
    echo "expected mode=$want_mode" >&2
    docker logs --since 30s "$CONTAINER" 2>&1 | grep '\[cohort\] gen=' | tail -16 || true
    return 1
  fi
  local g
  g=$(echo "$line" | grep -o 'gen=[0-9]*' | head -1 | cut -d= -f2)
  confirm_gen "$g"
}

health() {
  curl -sf -m 5 "$ENGINE/health" >/dev/null
}

echo "isolation_s11 start remain_node_s=$REMAIN_NODE_S estimate=$ESTIMATE_NODE_S"
health || { echo "engine not healthy" >&2; exit 5; }

# Neutralize then four arms. Always switch snapshot so helper.reset runs.
arm() {  # name mode quota gcap extra_args...
  local name=$1 mode=$2 quota=$3 gcap=$4
  shift 4
  budget_ok || { log_run "isolation_$name" fail 0 "budget"; exit 4; }
  local t0 t1
  t0=$(date +%s)
  echo "=== isolation arm $name ==="
  flush_cache
  # Dummy snapshot + ping so helper.reset is observed even if target==current.
  set_policy rotate 1 0 "$MEMBERS" "$MIX_CAP"
  ping_refresh
  confirm_mode rotate
  flush_cache
  set_policy "$mode" "$quota" "$gcap" "$MEMBERS" "$MIX_CAP"
  probe_empty "$name"
  confirm_mode "$mode"
  flush_cache
  probe_empty "$name"
  flush_cache
  python3 "$ROOT/isolation_client.py" \
    --label "$name" --teams 12 --members "$MEMBERS" \
    --rounds 5 --warmup-rounds 2 --gen-tokens 144 --context 9728 \
    --capacity "$MIX_CAP" --stagger 2.0 --seed "$SEED" \
    --reset-multiple 0 \
    --out "$OUT/mixed_${name}_12t.json" "$@"
  t1=$(date +%s)
  log_run "isolation_$name" ok "$((t1-t0))" "flush+no_filler"
}

# TeamCache first so it is not adjacent after Static-rate.
arm teamcache unified 0 0
arm lru group 0 0
arm static_rate group "$MIX_Q" 0 --rank-map "$ROOT/cpu/static_rate_mixed.json"
arm group_blind global 0 "$MIX_GCAP"

date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/DONE"
ELAPSED=$(( $(date +%s) - NODE_START ))
echo "DONE isolation_s11 node_elapsed=${ELAPSED}s gpu_hours=$(python3 -c "print(f'{$ELAPSED*8/3600:.4f}')")"
log_run isolation_done ok "$ELAPSED" ""
