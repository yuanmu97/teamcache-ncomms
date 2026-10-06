# Changelog

## 0.3.0

- Add an OpenAI-compatible gateway that automatically discovers team/member
  identity and injects SGLang request ids.
- Add a shared CPU-only discovery coordinator for multiple gateway replicas.
- Add rendezvous-hash team affinity, backend health tracking, streaming proxy
  support, tenant scoping, bearer authentication, and fail-open behavior.
- Reject malformed explicit request ids instead of silently replacing them,
  and use the coordinator's monotonic clock for expiry across gateway replicas.
- Scope idempotency records by tenant and model so equal request ids cannot
  cross isolation boundaries.
- Derive member turns from expiring lineage state instead of retaining an
  unbounded duplicate turn table in the coordinator.
- Report aggregate new-lineage and handoff counts for operational discovery
  validation without exposing prompts or team identities.
- Key cluster prompt fingerprints with HMAC when a discovery or dedicated
  fingerprint secret is configured, reducing offline dictionary exposure.
- Replace lineage-wide scans with cumulative-prefix, output-size, and stuffed-
  handoff indexes.
- Make online discovery two-phase so completions can be learned after a request
  has already been routed.
- Maintain policy token totals and active-team windows incrementally, reducing
  scheduler-side bookkeeping without changing admission decisions.
- Expose SGLang's in-flight request cap as `teamcache serve
  --max-running-requests` for physical active-KV pressure control.
- Allow the public A/B benchmark to reverse arm order, and publish the
  opposite-order 8×H200/GLM-5.3 validation record.
- Add cluster integration tests, deployment documentation, and CPU hot-path
  measurements.

## 0.2.0

- Initial standalone open-source package with reversible SGLang patching,
  unified policy, one-command serving, and the public ring benchmark.
