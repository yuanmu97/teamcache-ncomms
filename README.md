# Reproduction code for "Collaboration topology governs inference-cache reuse in language-model agent teams"

Mu Yuan, Jinke Song, Junyue Liu, Siyang Jiang, Junyang Wang, Zhaomeng Zhou, Lan Zhang, Yunhao Liu, Guoliang Xing

Code DOI: `10.5281/zenodo.YYYYYYY` · Data DOI: `10.5281/zenodo.XXXXXXX`

This repository regenerates every figure, Source Data file and quoted number of the paper and its Supplementary Information from the archived data, and reruns all cache-level experiments, on an ordinary CPU.
It also contains the scripts that ran the GPU experiments on an 8×B300-GPU server and their repetitions on a 4×A100 server, and, in `teamcache/`, the installable TeamCache package for SGLang with the gateway that names requests by team and member.

## System requirements

- Operating system: macOS or Linux; tested on macOS 26 (Apple M3, 8 cores, 16 GB), and run on Ubuntu by the continuous-integration workflow in `.github/workflows/ci.yml`.
- Python 3.9–3.12; tested with Python 3.9.6 and 3.11.
- Python packages, pinned in `requirements.txt`: numpy 2.0.2, matplotlib 3.9.4, pandas 2.3.3, openpyxl 3.1.5, pyarrow 21.0.0, pillow 11.3.0.
- No GPU or other non-standard hardware is needed for anything except the GPU experiments in `gpu_experiments/`.
- Disk: 115 MB for the data deposit, 200 MB more for the optional public traces.
- Figures use Arial when it is installed and DejaVu Sans otherwise; the font changes glyph shapes only, never values.

## Installation

```bash
git clone https://github.com/yuanmu97/teamcache-ncomms
cd teamcache-ncomms
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # under 1 min
make data                                # downloads and verifies the Zenodo deposit into ./data
```

`conda env create -f environment.yml` works as well.
Without network access, place `teamcache-ncomms-data-v1.0.0.zip` anywhere and run `TEAMCACHE_DATA_ZIP=/path/to/zip make data`.

## Demo

```bash
make demo
```

This rebuilds the per-run GPU tables from the raw GPU result files, compares them with the deposited tables, and recomputes every number quoted in the paper from the data.
Expected output ends with

```
identical  mixed_rows.json
...
252 of 255 claims reproduced; 1 reproduced by command; 2 by GPU probe; 0 failed
```

and the full claim table is written to `out/CLAIMS.csv`.
Run time: a few seconds on an Apple M3.

## Reproducing the paper

| Command | What it does | Time (Apple M3, 8 cores) |
| --- | --- | --- |
| `make check` | recomputes the 255 quoted numbers listed in `CLAIMS.csv` | 1 s |
| `make figures` | regenerates Figs. 1–8, Supplementary Figs. 1–7 and the Source Data files into `out/` | 10 s |
| `make cache-level` | reruns every cache-level experiment into `out/cache_level/` and compares with the deposit | 40 s |
| `make revision` | reruns the block-level, sensitivity, shadow-estimator, group-aware and Discovery analyses (Supplementary Notes 3–6 and 12) | 1.5 min |
| `make check-cache-level` | recomputes the quoted numbers from the rerun experiments | 1 s |
| `make figures-from-scratch` | cache-level experiments, then figures from their outputs | 1 min |
| `make public-data public-traces` | downloads MAST and Who&When at pinned revisions and reruns the public-trace experiment | 10 s + download (200 MB) |

All cache-level experiments are deterministic: they use fixed seeds, and their outputs are identical to the deposited ones.
`make cache-level` prints `identical` for each output file.
Without the public traces, the public-trace experiment is read from the deposit.

Environment variables select other locations: `TEAMCACHE_DATA` (deposit, default `./data`), `TEAMCACHE_OUT` (outputs, default `./out`), `TEAMCACHE_PUBLIC` (public traces) and `TEAMCACHE_CACHE_LEVEL` (which cache-level results the figures read).

### Where each figure comes from

| Figure | Function in `figures/make_figures.py` | Inputs |
| --- | --- | --- |
| Fig. 1 | `fig1` | `gpu/results/collections-summary.json`, `cache_level/s8_public_traces.json` |
| Fig. 2 | `fig2` | `gpu/results/team-sweep/`, GLM ring runs, `cache_level/s2_cliff.json`, `cache_level/s3_phase.json` |
| Fig. 3 | `fig3` | `cache_level/s5_policy_trace.json` |
| Fig. 4 | `fig4` | `cache_level/s3_phase.json`, `cache_level/s6_baselines.json`, `cache_level/s1_grid.json`, `cache_level/s14_group_baselines.json`, `cache_level/s8_public_traces.json` |
| Fig. 5 | `fig5` | ring, selector and mixed GPU runs; A100 repetitions via `teamcache_ncomms/a100.py` |
| Fig. 6 | `fig6` | mixed GPU runs (`gpu/mixed/`) |
| Fig. 7 | `fig7` | `gpu/figure_data/sweep_rows.json` |
| Fig. 8 | `fig8` | HiCache, GLM 5.3 and vLLM runs |
| Supplementary Figs. 1 and 2 | `si_ring_cliff`, `si_phase_detail` | cache-level outputs |
| Supplementary Fig. 3 | `si_discovery` | schematic, no data |
| Supplementary Figs. 4–6 | `si_heatmaps`, `si_frontier_and_cost`, `si_ring_turns` | cache-level outputs, B300 and A100 ring GPU runs |
| Supplementary Fig. 7 | `si_star` | B300 and A100 star GPU runs |

### Cache-level experiments

| Module | Output | Paper |
| --- | --- | --- |
| `teamcache_ncomms.run_sims s1` | `s1_grid.json` | Fig. 4c, Supplementary Figs. 4 and 5 |
| `teamcache_ncomms.run_sims s2` | `s2_cliff.json` | Fig. 2d, Supplementary Fig. 1 |
| `teamcache_ncomms.run_sims s3` | `s3_phase.json` | Figs. 2e and 4a, Supplementary Fig. 2 |
| `teamcache_ncomms.run_sims s4` | `s4_cost.json` | Supplementary Fig. 5 |
| `teamcache_ncomms.policy_trace` | `s5_policy_trace.json` | Fig. 3b–d |
| `teamcache_ncomms.run_baselines` | `s6_baselines.json` | Fig. 4b,c |
| `teamcache_ncomms.trace_replay` | `s7_trace_replay.json` | replay of the recorded GPU order |
| `teamcache_ncomms.public_traces` | `s8_public_traces.json` | Figs. 1b and 4d |
| `teamcache_ncomms.regime` | `s10_regime.json` | supporting analysis |
| `teamcache_ncomms.optimal sensitivity` | printed table | Supplementary Note 15 (forced insertion) |
| `teamcache_ncomms.optimal waits-ablation` | printed table | Supplementary Note 15 (wait-graph alternative) |
| `teamcache_ncomms.block_sim` | `s11_block_level.json` | Supplementary Note 5 |
| `teamcache_ncomms.block_sim --sensitivity` | `s13_sensitivity.json` | Supplementary Note 6 |
| `teamcache_ncomms.shadow_error` | `s12_shadow_error.json` | Supplementary Note 3 |
| `teamcache_ncomms.group_baselines` | `s14_group_baselines.json` | Fig. 4c, Supplementary Note 4 |
| `teamcache_ncomms.discovery_eval` | `s15_discovery.json` | Supplementary Note 12 |

`teamcache_ncomms/optimal.py` holds the cache model, the TeamCache admission rule and the fixed baselines; `teamcache_ncomms/classic.py` holds ARC, LIRS, 2Q, S3-FIFO and W-TinyLFU.

## GPU experiments

The GPU measurements were taken on an 8×B300-GPU server with Kimi K3 and GLM 5.3 served by SGLang and vLLM.
`gpu_experiments/README.md` lists the container images by digest, the engine patches, and the driver script behind every deposited GPU result.
Rerunning them needs the same class of hardware and model weights; reproducing the figures does not.

The repetitions of Supplementary Note 7 ran on a 4×A100-80GB server with GLM-4.7-Flash, SGLang 0.5.18 and vLLM 0.30.0, using the same engine patch, gateway and clients.
Their drivers are in `gpu_experiments/runs/a100/`, and `python -m teamcache_ncomms.a100` summarises `data/gpu_a100/` into `out/gpu_a100_summary.json`, from which `make check` computes the quoted A100 numbers.

Two quoted measurements, the 61,440-token cold-prefill time and the throughput of independent requests, are taken directly on the GPU server by the probe scripts that `CLAIMS.csv` names.

## TeamCache package

`teamcache/` is a self-contained Python package that patches SGLang's cache-insertion sites, launches a model server and runs the gateway with Team Discovery and team-affinity routing.
It is installed separately with `pip install -e teamcache` and needs a GPU only for serving; `teamcache/README.md` describes installation, request marking, cluster deployment and an included A/B benchmark.

## Repository layout

```
teamcache_ncomms/   cache model, policies, cache-level experiments, GPU tables, claim check
figures/            figure and Source Data scripts
gpu_experiments/    engine patches, recorders, replayers and run scripts used on the GPU server
scripts/            data download scripts
teamcache/          installable TeamCache package for SGLang (gateway, Team Discovery, admission policy)
```

## Licence and citation

Code: MIT (`LICENSE`).
Data: CC BY 4.0 (see the deposit).
Please cite the paper; `CITATION.cff` gives the paper, this code and the data deposit.
