# Ring 20×5 benchmark

This is the single public case selected to demonstrate TeamCache's intended
operating point without shipping private traces or the full paper artifact.

The canonical dataset specification lives at
`src/teamcache/data/ring_20x5.json` and is included in built wheels. Prompts are
generated from `(seed, team, member, step)` at run time. Each member begins with
a private 3,000-token prefix, then receives the same growing team transcript.
The speaking order is round robin within each team, while all 20 teams run
concurrently.

This preserves the property the mechanism targets: a member's next prompt
strictly extends its prior prompt, but five members take turns and the combined
working set exceeds the prefix pool.

Run the full A/B:

```bash
teamcache benchmark --endpoint http://127.0.0.1:30000 \
  --output ring-20x5-ab.json
```

Isolation rules enforced by the runner:

1. both arms use the same patched engine process;
2. only `mode=off` versus `mode=unified` differs; the pool budget is unchanged;
3. the engine's `/flush_cache` endpoint is called between arms;
4. prompt ids and generated input token ids are identical across arms;
5. the reported steady window is rounds 5–8.

Use `--order teamcache-first` to reverse the default execution order when
checking for server warm-up bias.

`reported.json` contains the compact, three-seed result from the paper machine.
`h200_glm53_420800.json` records two full runs in opposite arm orders on
8×H200. `bmax_glm47_180k.json` records a cross-model validation on 4×A100 after
calibrating a 180k-token physical pool to the same near-zero-hit LRU pressure
regime. It keeps the complete input geometry but generates one token per
request, so it validates input-prefix caching rather than end-to-end decode.
These files are evidence, not golden tests: wall time is hardware-specific.
