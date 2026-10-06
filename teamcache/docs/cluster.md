# Cluster deployment

TeamCache's cache decision is local to an SGLang process, but team identity
must be global. The cluster path therefore separates three roles:

```text
OpenAI clients
      │
      ▼
TeamCache gateway replicas ─────► discovery coordinator (CPU only)
      │                            hashes and lengths only
      │ rendezvous hash(team_id)
      ├────────► SGLang GPU node 0 + TeamCache
      ├────────► SGLang GPU node 1 + TeamCache
      └────────► SGLang GPU node N + TeamCache
```

The shared coordinator assigns lineage, team, member, and turn identifiers.
Every gateway independently reaches the same backend choice from the returned
team id. Adding or removing a backend remaps only the teams whose rendezvous
winner changed.

## 1. Start the GPU nodes

Install TeamCache and SGLang on each GPU node. Nodes may use different tensor
parallel sizes, but a homogeneous KV token budget is recommended so a team's
behavior does not depend on which node owns it.

```bash
teamcache serve \
  --model-path /models/your-model \
  --tp-size 8 \
  --pool-tokens 420800 \
  --host 0.0.0.0 \
  --port 30000 \
  -- --trust-remote-code
```

Repeat on every node. Confirm each node's `/health` endpoint before adding it
to the gateway.

## 2. Start the discovery coordinator

Run one coordinator reachable by all gateway replicas. It needs CPU and memory,
not a GPU.

```bash
export TEAMCACHE_DISCOVERY_TOKEN='a-long-random-secret'
teamcache discovery \
  --host 0.0.0.0 \
  --port 39090 \
  --window-s 7200
```

Protect this internal endpoint with a service mesh, private network, or network
policy. `TEAMCACHE_DISCOVERY_TOKEN` enables bearer-token authentication; use
the same environment variable on gateways. Gateways also use that token as an
HMAC key for prompt fingerprints, preventing the coordinator from testing
plain SHA-256 guesses for common prompts. Set `TEAMCACHE_FINGERPRINT_KEY` on
all gateways to use a separate HMAC key. Without either variable, fingerprints
remain deterministic SHA-256 values for zero-configuration deployments.

The coordinator receives cumulative SHA-256 fingerprints, role names, message
lengths, model name, and tenant scope. It never receives prompt or completion
text. `/health` is a liveness endpoint and `/stats` reports counts only.
The latter includes `identified`, `completed`, `new_lineages`, and `handoffs`
so operators can validate learning without logging team identities.

## 3. Start one or more gateways

Every replica must use the same `--discovery-url` and the same backend list.

```bash
export TEAMCACHE_DISCOVERY_TOKEN='a-long-random-secret'
teamcache gateway \
  --host 0.0.0.0 \
  --port 8000 \
  --discovery-url http://teamcache-discovery:39090 \
  --backend http://sglang-0:30000 \
  --backend http://sglang-1:30000 \
  --backend http://sglang-2:30000
```

Point an unmodified OpenAI client at `http://gateway:8000/v1`. Both ordinary
JSON responses and server-sent-event streaming are proxied. The gateway adds:

- `x-teamcache-discovery`: `explicit`, `discovered`, or `bypass`;
- `x-teamcache-backend`: the selected backend URL.

These response headers make routing observable without logging prompt text.
Automatic discovery currently targets OpenAI-compatible requests with a
`messages` array. Native `/generate` calls that contain only token ids must
provide an explicit `rid`, because message-boundary lineage cannot be inferred
from a flat token array without tokenizer-specific parsing.

## Tenant isolation

Discovery is always scoped by tenant and model. In a multi-tenant deployment,
the trusted ingress should set `x-teamcache-tenant`; do not accept that header
directly from untrusted users. Change the name with `--tenant-header` if the
platform already has a canonical tenant header.

For a single-tenant service, `--default-tenant` is sufficient.

## Explicit metadata fast path

If the application already knows team membership, send the normal SGLang
`rid` and discovery is skipped:

```json
{"rid":"research-42:3:8","model":"default","messages":[...]}
```

The gateway also accepts a convenience object and removes it before forwarding:

```json
{
  "teamcache":{"team":"research-42","member":3,"turn":8},
  "model":"default",
  "messages":[...]
}
```

## Failure behavior

- Discovery is fail-open by default: if the coordinator is unavailable, the
  request is forwarded without a TeamCache id and SGLang uses upstream LRU.
  Use `--discovery-required` when silent fallback is undesirable.
- Gateways probe every backend's `/health` endpoint. Unhealthy nodes are
  removed from the rendezvous candidate set; teams are deterministically
  remapped over the remaining nodes. Configure the interval with
  `--health-interval`.
- The gateway does not automatically replay a request after an ambiguous
  upstream failure, because doing so can duplicate model side effects.
- A coordinator restart is safe but starts a new discovery window. Its default
  request-id namespace contains a random restart epoch, preventing newly
  inferred teams from colliding with policy state left by the old process.
  Teams are relearned and may move once; explicit request ids keep their normal
  routing. Use `--namespace` only when the operator controls state resets.

## Capacity planning

Discovery lookup is indexed by cumulative prompt fingerprint rather than by a
scan over all live lineages. State expires after `--window-s`. The default
100,000 idempotency records can be changed with `--max-requests`.

Use one coordinator per independent serving fleet. If a deployment needs
failure-independent discovery authorities, place the coordinator under an
orchestrator restart policy; active-active replicated state is not implemented
in version 0.3.
