# GPU experiments

These are the scripts that produced the GPU measurements in the data deposit, kept as they were run.
They need an 8×B300-GPU server (AWS p6-b300.48xlarge in the paper), Docker with the NVIDIA runtime, and the Kimi K3 and GLM 5.3 weights.
Paths such as `/opt/dlami/nvme/pat` are the working directory on that server; set `PAT`, `NVME` and the model paths in `engine/common_b300.sh` for another host.
None of these scripts is needed to reproduce the figures, which are rebuilt from the deposited results on a CPU.

## Software

| Component | Version |
| --- | --- |
| SGLang (Kimi K3) | `lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9` |
| SGLang (GLM 5.3) | `lmsysorg/sglang@sha256:710bc11443a7b1807d69803386468101bcfced35f86bf8fe92a8209e05a2f052` |
| vLLM (Kimi K3) | `vllm/vllm-openai@sha256:969a492a96dd590b41a384dc610175814d7dbda3362d8b2553659cf11e523a7c` |
| AutoGen, CrewAI | versions recorded in the `user-agent` of the original traces; harnesses in `recording/` |

## Layout

| Directory | Content |
| --- | --- |
| `engine/` | Engine launch scripts and the patch generators. `patch_sglang_cohort.py` writes a patched `mem_cache/common.py` that is bind-mounted into the SGLang container; `patch_vllm_cohort.py` does the same for vLLM's `BlockPool.cache_full_blocks`. The policy is switched at run time through `/tmp/cohort/control.json`, so all policies of one comparison share one engine boot. |
| `recording/` | Recording proxies (`proxy.py` for Anthropic Messages, `proxy_openai.py` for OpenAI chat completions) and the AutoGen, CrewAI and selector harnesses that produced `traces/` |
| `replay/` | Closed-loop replayers that reproduce member order, team membership, prompt lengths and wait dependencies of a trace against the engine |
| `runs/` | Drivers of each experiment (below); `runs/mixed/` holds the mixed-workload driver with its frozen `plan.json` |
| `probes/` | Single-purpose measurements (cold-prefill cost, HiCache cost, capacity, throughput non-regression) |
| `analysis/` | Scoring and analysis, including `team_discovery.py` (Supplementary Table 1) and `phase0_cpu.py` (admission overhead) |

## Experiment → script → deposited result

| Paper | Driver | Result in the deposit |
| --- | --- | --- |
| Fig. 2a (Kimi K3), Fig. 7 | `runs/sweep_team.sh` | `gpu/results/team-sweep/` |
| Fig. 2a (GLM 5.3) | `runs/run_glm_ring_lru.sh` | `gpu/results/e2e/e2e-ring-lru-*-glm.json` |
| Fig. 5a, Supplementary Fig. 6 | `runs/e2e_autogen.sh` (ring, three seeds) | `gpu/results/e2e/e2e-autogen-design-*-20t-p3000r8*.json` |
| Fig. 5b | `runs/run_selector_standaside.sh` | `gpu/results/e2e/e2e-selector-debug-*-20t-p3000r8.json` |
| Figs. 5c,d and 6 | `runs/mixed/run_gpu.sh` (seeds 11–14) | `gpu/mixed/mixed_*_12t*.json` |
| Supplementary Note 14 | `runs/mixed/isolation_run.sh` | `gpu/mixed/isolation_s11/` |
| Fig. 8a | `runs/run_hicache.sh` | `gpu/results/e2e/e2e-autogen-design-*-hicache.json` |
| Fig. 8b | `runs/run_hicache_fleet.sh`, `runs/run_hicache_oversub.sh` | `gpu/results/e2e/e2e-fleet-mixed-*-10t-hicache-joint.json` |
| Fig. 8c,d | `runs/run_glm_scale.sh` | `gpu/results/e2e/e2e-fleet-mixed-*-32t-glm-scale.json` |
| Fig. 8e, Supplementary Note 8, Supplementary Table 7 | `runs/run_vllm_q9.sh` | `gpu/results/vllm/` |
| Supplementary Note 10, Supplementary Table 9 | `runs/run_longctx_field.sh` | `gpu/results/e2e/e2e-longctx-*.json` |
| Supplementary Table 12, Supplementary Fig. 7 | `runs/sweep_ablate.sh`, `runs/mixed/run_gpu.sh` (star) | `gpu/results/team-ablate/`, `gpu/mixed/star_*` |
| Admission overhead | `analysis/phase0_cpu.py` (host CPU) | `cpu/phase0_cpu.json` |
| Cold prefill 2.79 s vs 0.13 s | `probes/measure_cost_model.py` | printed by the probe |
| Throughput within ±0.2% | `probes/probe_nonregress.py` (run with the policy off and on) | printed by the probe |

## Repetitions on 4×A100 (Supplementary Note 7)

The A100 campaign ran GLM-4.7-Flash (BF16) on a 4×A100-SXM4-80GB server with SGLang 0.5.18 and vLLM 0.30.0 installed from PyPI, with the same `patch_sglang_cohort.py`, `patch_vllm_cohort.py`, gateway and clients as above.
The drivers in `runs/a100/` expect the scripts of this directory flattened into one working directory `TC`; set `TC`, `VENV` and `MODEL` in `runs/a100/common_a100.sh`.

| Paper | Driver | Result in the deposit |
| --- | --- | --- |
| Throughput of independent requests | `campaign.sh` (phase `e8`) | `gpu_a100/nonregress/` |
| Fig. 6d, live ring and selector | `campaign.sh` (phases `ring`, `selector`), `e2e_autogen_a100.sh` | `gpu_a100/e2e/`, gateway traces in `gpu_a100/traces/` |
| Fig. 6d, mixed workload; star | `campaign.sh` (phases `mixed`, `star`) with `runs/mixed/` | `gpu_a100/mixed/`, `gpu_a100/star/` |
| Capacity sweep | `campaign.sh` (phase `sweep`) | `gpu_a100/team-sweep/` |
| HiCache ring and long-context fleet | `campaign_ext.sh` (`hicache_ring`, `hicache_fleet` with `POOL=210400` and the halved `FLEET_MIX`) | `gpu_a100/hicache/` |
| vLLM | `vllm_a100.sh`, `e2e_autogen_vllm.sh` | `gpu_a100/vllm/` |

The selector and vLLM runs set `PROXY_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'`, which the gateway forwards as `chat_template_kwargs`; with thinking on, GLM-4.7-Flash spent the 144-token cap on reasoning.
`gpu_a100/runs.csv` logs every run in order.

A typical sequence for one comparison:

```bash
python3 engine/patch_sglang_cohort.py --out /tmp/sgl-patch
engine/serve_kimi_k3_b300_sglang_cohort.sh      # boots the engine with the patched cache
engine/wait_engine.sh                            # waits until the engine reports healthy
runs/sweep_team.sh                               # writes one JSON per policy and team count
```

The TeamCache package used for the patches is in `../teamcache/`.
