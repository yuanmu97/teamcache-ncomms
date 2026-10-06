#!/usr/bin/env bash
# Shared paths for the 8xB300 p6-b300.48xlarge instance.
set -euo pipefail

export NVME=/opt/dlami/nvme
export MODEL_GLM53=$NVME/models/GLM-5.3
export MODEL_KIMI_K3=${MODEL_KIMI_K3:-$NVME/models/Kimi-K3}
export MODEL_KIMI_K3_DSPARK=${MODEL_KIMI_K3_DSPARK:-$NVME/models/Kimi-K3-DSpark}
# Local immutable tag created from:
# lmsysorg/sglang@sha256:710bc11443a7b1807d69803386468101bcfced35f86bf8fe92a8209e05a2f052
export SGLANG_B300_IMAGE=${SGLANG_B300_IMAGE:-sglang-b300:cu130-710bc114}
# lmsysorg/sglang@sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9
export SGLANG_KIMI_K3_IMAGE=${SGLANG_KIMI_K3_IMAGE:-sglang-kimi-k3:d6e7288627be}
# vllm/vllm-openai@sha256:969a492a96dd590b41a384dc610175814d7dbda3362d8b2553659cf11e523a7c
export VLLM_KIMI_K3_IMAGE=${VLLM_KIMI_K3_IMAGE:-vllm-kimi-k3:5894fdf}
export SGLANG_PORT=${SGLANG_PORT:-30000}
export LOGDIR=$NVME/logs
export B300_MACHINE_USD_PER_H=${B300_MACHINE_USD_PER_H:-112.32}
