PYTHON ?= python3
export PYTHONPATH := $(CURDIR)
export TEAMCACHE_DATA ?= $(CURDIR)/data
export TEAMCACHE_OUT ?= $(CURDIR)/out
export TEAMCACHE_PUBLIC ?= $(TEAMCACHE_DATA)/public_raw

CACHE_LEVEL_STEPS = "run_sims s1" "run_sims s2" "run_sims s3" "run_sims s4" \
                    run_baselines policy_trace trace_replay regime

.PHONY: help data public-data demo check tables cache-level revision public-traces \
        check-cache-level figures figures-from-scratch all gpu clean

help:
	@echo "make data                 download and verify the Zenodo data deposit into ./data"
	@echo "make demo                 rebuild the GPU tables and recompute every quoted number (~10 s)"
	@echo "make check                recompute every number quoted in the paper from ./data"
	@echo "make tables               rebuild per-run GPU tables from raw GPU results and compare"
	@echo "make cache-level          rerun all cache-level experiments into ./out and compare"
	@echo "make revision             block-level, sensitivity, shadow-error, group-aware and Discovery analyses"
	@echo "make public-data          download the public MAST and Who&When traces"
	@echo "make public-traces        rerun the public-trace experiment (needs public-data)"
	@echo "make check-cache-level    recompute the quoted numbers from the rerun experiments"
	@echo "make figures              regenerate Figs. 1-8, Supplementary Figs. and Source Data"
	@echo "make figures-from-scratch cache-level experiments, then figures from their outputs"
	@echo "make all                  everything above except the public traces"
	@echo "make gpu                  how to rerun the GPU experiments (needs an 8xB300-GPU server)"

data:
	bash scripts/fetch_data.sh

public-data:
	bash scripts/fetch_public_traces.sh

demo: tables check

check:
	$(PYTHON) -m teamcache_ncomms.claims --csv $(TEAMCACHE_OUT)/CLAIMS.csv

tables:
	$(PYTHON) -m teamcache_ncomms.gpu_tables
	$(PYTHON) -m teamcache_ncomms.collections_summary > /dev/null
	$(PYTHON) -m teamcache_ncomms.compare figure-data

cache-level:
	@for step in $(CACHE_LEVEL_STEPS); do \
	  echo "== $$step"; $(PYTHON) -m teamcache_ncomms.$$step > /dev/null || exit 1; \
	done
	$(PYTHON) -m teamcache_ncomms.compare cache-level

A100 = $(TEAMCACHE_DATA)/gpu_a100
SHADOW_EXTRA ?= --extra "A100 ring=$(A100)/traces/e2e-autogen-design-lru-20t-p3000r8-a100.jsonl" \
                --extra "A100 ring=$(A100)/traces/e2e-autogen-design-lru-20t-p3000r8-a100-s[0-9].jsonl" \
                --extra "A100 selector=$(A100)/traces/e2e-selector-debug-lru-20t-p3000r8-a100*.jsonl" \
                --extra "A100 mixed=$(A100)/mixed/mixed_lru_12t_s*.requests.jsonl"

revision:
	$(PYTHON) -m teamcache_ncomms.block_sim
	$(PYTHON) -m teamcache_ncomms.block_sim --sensitivity
	$(PYTHON) -m teamcache_ncomms.shadow_error $(SHADOW_EXTRA)
	$(PYTHON) -m teamcache_ncomms.group_baselines
	$(PYTHON) -m teamcache_ncomms.discovery_eval

public-traces:
	$(PYTHON) -m teamcache_ncomms.public_traces
	$(PYTHON) -m teamcache_ncomms.compare cache-level

check-cache-level:
	TEAMCACHE_CACHE_LEVEL=$(TEAMCACHE_OUT)/cache_level $(PYTHON) -m teamcache_ncomms.claims

figures:
	$(PYTHON) figures/make_figures.py

figures-from-scratch: cache-level
	TEAMCACHE_CACHE_LEVEL=$(TEAMCACHE_OUT)/cache_level $(MAKE) figures

all: demo cache-level check-cache-level figures

gpu:
	@sed -n '/^## GPU experiments/,/^## /p' README.md | sed '$$d'

clean:
	rm -rf $(TEAMCACHE_OUT)
