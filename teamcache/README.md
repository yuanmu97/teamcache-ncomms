# TeamCache

Team-aware prefix-cache admission for serving concurrent LLM agent teams with
[SGLang](https://github.com/sgl-project/sglang).

TeamCache is a small engine hook, not an inference-engine fork. It keeps SGLang's
normal prefix matching, scheduling, and LRU eviction, and only decides whether a
completed request should enter the radix cache. On round-robin teams, this stops
the member that just spoke from evicting the member that speaks next.

This directory is the installable TeamCache package, for **using the system**.
It contains:

- a reversible SGLang patch and one-command model launcher;
- the evaluated `unified` TeamCache policy;
- an OpenAI-compatible gateway with automatic team discovery and cluster-wide
  team-affinity routing;
- a privacy-safe, procedurally generated 20-team benchmark case;
- documented LRU-vs-TeamCache results, plus an A/B runner.

The rest of this repository reproduces the figures and numbers of the paper
*Collaboration topology governs inference-cache reuse in language-model agent
teams* (Nature Communications) from its data deposit
(DOI [10.5281/zenodo.23187430](https://doi.org/10.5281/zenodo.23187430)); see the
[repository README](../README.md).

## Quick start

Prerequisites: Linux, Python 3.9+, NVIDIA GPUs, and a working SGLang installation.
First verify that upstream SGLang can load your model.

```bash
cd teamcache
pip install -e .
teamcache selftest
teamcache serve \
  --model-path /path/to/model \
  --tp-size 8 \
  --pool-tokens 420800 \
  -- --trust-remote-code
```

That one `serve` command:

1. patches the installed SGLang radix-insert sites and keeps a
   `.teamcache.bak` backup;
2. writes the TeamCache control file;
3. launches an OpenAI-compatible SGLang server on port 30000.

`--pool-tokens` is passed to SGLang as `--max-total-tokens` and used as the
policy budget, so the engine and TeamCache use the same capacity. Model-specific
SGLang flags go after `--`.

If concurrent in-flight prompts consume nearly the whole pool before any
completed prefix can remain cached, set `--max-running-requests` to an explicit
capacity. This is a serving-capacity control shared by both A/B arms, not a
TeamCache-only optimization.

To preview the launch without starting the model, add `--dry-run`.

## Mark requests

TeamCache reads the request id SGLang already carries. Use
`team:member:turn`, where `member` is an integer. Traffic without this shape
continues to use upstream LRU.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")
client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Continue the analysis."}],
    extra_body={"rid": "team-alpha:2:7"},
)
```

Native `/generate` requests can set the same value in the top-level `rid`
field. See [`examples/set_request_id.py`](examples/set_request_id.py).

## Automatic discovery and GPU clusters

Applications that do not expose team/member metadata can point their normal
OpenAI client at the TeamCache gateway. The gateway fingerprints message
prefixes locally, obtains a cluster-wide identity, injects the request id, and
uses rendezvous hashing to keep every discovered team on one SGLang backend.
Raw prompt text is never sent to the discovery coordinator.

When `TEAMCACHE_DISCOVERY_TOKEN` is set, gateway fingerprints are HMAC-keyed
with it as well as authenticated in transit; a separate
`TEAMCACHE_FINGERPRINT_KEY` can be used if preferred.

For a single gateway:

```bash
teamcache gateway --port 8000 \
  --backend http://gpu-0:30000 \
  --backend http://gpu-1:30000
```

For multiple gateway replicas, run one shared coordinator and point every
replica at it:

```bash
export TEAMCACHE_DISCOVERY_TOKEN='replace-with-a-secret'
teamcache discovery --host 0.0.0.0 --port 39090

teamcache gateway --port 8000 \
  --discovery-url http://discovery:39090 \
  --backend http://gpu-0:30000 \
  --backend http://gpu-1:30000
```

Then change the application's OpenAI base URL to the gateway, for example
`http://gateway:8000/v1`. Existing valid `rid` values bypass inference and
still receive team-affinity routing. See the full deployment and failure-mode
guide in [`docs/cluster.md`](docs/cluster.md).

### Control-plane overhead

Version 0.3 replaces lineage-wide scans with cumulative-prefix indexes and
maintains policy capacity counters incrementally. A local developer
microbenchmark measured a 38.7× discovery lookup improvement at 2,000 live
lineages and a 2.87× policy-predicate improvement at 200 active teams, with
identical routing/admission decisions. These CPU-only numbers are documented in
[`benchmarks/cpu_hot_path.json`](benchmarks/cpu_hot_path.json) and are not
intended as portable latency guarantees.

## Run the included A/B case

With the server still running:

```bash
teamcache benchmark --endpoint http://127.0.0.1:30000 \
  --output teamcache-ab.json
```

The runner applies two policies to the same patched engine, physically flushes
the prefix cache between arms, and sends byte-identical workloads:

- **LRU:** TeamCache disabled (`mode=off`) with the same pool budget;
- **TeamCache:** the default `unified` policy.

For a short connectivity check, add `--quick`; those numbers are not a
performance result.

For an order-bias check, repeat the full run with `--order teamcache-first`.
The runner still flushes the radix cache before each arm.

To validate cache behavior with the complete input workload while minimizing
decode cost, add `--generation-tokens 1`. This preserves all 20×5×8 request
prefixes and their cache pressure, but it is an input-only systems check rather
than the paper's end-to-end latency benchmark.

When calibrating an equivalent KV-cache pressure point on different hardware
or a different model, use `--arm lru` first and adjust the server's
`--pool-tokens`; pass the same value to `benchmark --pool-tokens`, then run the
default two-arm command at the selected capacity.

Calibrate the *retained-prefix* budget, not just `max-total-tokens`: KV for
running requests comes from the same pool. If the engine admits many more
simultaneous requests than the workload's team count, cap
`--max-running-requests` consistently for both arms. For exact admission
bypass, keep `--chunked-prefill-size` above the case's longest prompt; an
unfinished multi-chunk prefill must remain accounted by SGLang even when its
completed request would later be bypassed.

The bundled dataset is
[`src/teamcache/data/ring_20x5.json`](src/teamcache/data/ring_20x5.json). It is a
CC0 procedural specification rather than redistributed conversation text: 20
concurrent teams, five round-robin members per team, eight rounds, a private
3,000-token preamble, and 144 new transcript tokens per step. The generator is
deterministic and sends token ids directly, making the case free of private or
model-generated text. Set `--vocab-size` if your model has fewer than 32,000
tokens.

### Reported result

The paper run used the same 20×5×8 geometry on one 8×B300 node with Kimi K3 and
the SGLang image recorded in
[`benchmarks/ring_20x5/reported.json`](benchmarks/ring_20x5/reported.json).
Means over three seeds:

| Policy | Steady-state prefix hit rate | Steady-state wall time |
| --- | ---: | ---: |
| SGLang LRU | 1.23% | 185.4 s |
| TeamCache | 35.52% | 150.1 s |

That is a **+34.29 percentage-point** hit-rate gain and **1.24×** steady-state
speedup. Hardware, model, engine build, and workload shape affect absolute
numbers; compare the two arms on the same server rather than expecting these
times on different hardware.

### Cross-model tight-cache validation

We also calibrated the physical KV pool on 4×A100 with GLM-4.7-Flash until
upstream LRU reached the same near-zero-hit pressure regime. With a 180,000
token pool and the full 20×5×8 input workload, LRU achieved 0.00% steady hits
in 317.6 s, while TeamCache achieved 18.05% in 299.2 s: **+18.05 points** and
**1.06× wall-time speedup**. Generation was limited to one token, so this is an
input-cache validation rather than an end-to-end generation result. The exact
record is in
[`benchmarks/ring_20x5/bmax_glm47_180k.json`](benchmarks/ring_20x5/bmax_glm47_180k.json).

On 8×H200 with GLM-5.3, a 420,800-token pool, 20 running requests, and a
single-chunk prefill, two trials in opposite arm orders measured 19.60% mean
steady TeamCache hits versus 0.00% for LRU. Mean wall times were 286.6 s versus
329.6 s, for **1.15× mean speedup** (range 1.141×–1.159×). The exact two-trial
record and immutable engine image id are in
[`benchmarks/ring_20x5/h200_glm53_420800.json`](benchmarks/ring_20x5/h200_glm53_420800.json).

## Existing SGLang deployments

You can patch without using the launcher:

```bash
teamcache install
export COHORT_POOL_TOKENS=420800
python -m sglang.launch_server --model-path /path/to/model ...
```

Or generate a file for a container bind mount:

```bash
teamcache install --out /tmp/teamcache/common.py
```

The control file is hot-reloaded at most once per second. A minimal file is:

```json
{"mode": "unified", "pool_tokens": 420800}
```

Set `COHORT_CONTROL=/path/to/control.json` in the SGLang process. The `COHORT_*`
name is retained internally for compatibility with the evaluated artifact.

## Safety and compatibility

- `teamcache install` fails closed if the expected SGLang edit sites are not
  found exactly once. It never guesses after an engine upgrade.
- `teamcache uninstall` restores the backup. Restart SGLang after install or
  uninstall.
- Bypass is exact when a prompt fits in one prefill chunk. Chunked unfinished
  prefills keep SGLang's required accounting and may retain completed chunks.
- The supported release path is SGLang. The vLLM prototype used in the paper's
  experiments is in [`../gpu_experiments/engine/`](../gpu_experiments/engine/) and
  is deliberately not packaged here because its cache-block semantics produced a
  much smaller and version-sensitive effect.
- Installing the engine hook alone does not infer team identity. Put
  `teamcache gateway` in front of SGLang for automatic discovery, or use the
  explicit request-id path when the application already knows membership.

## Development

```bash
pip install -e '.[dev]'
pytest -q
ruff check src tests
```

The CPU-only tests validate policy reductions, hot reload, patch installation,
and the procedural dataset. No model weights or GPU are required.

## Citation and license

TeamCache is MIT licensed. The procedural benchmark is CC0. Citation metadata
is in the repository's [`CITATION.cff`](../CITATION.cff); please cite the Nature
Communications paper *Collaboration topology governs inference-cache reuse in
language-model agent teams* (Yuan et al.). The repository, including this
package, is archived as
[10.5281/zenodo.23188240](https://doi.org/10.5281/zenodo.23188240).
