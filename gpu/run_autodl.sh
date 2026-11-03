#!/usr/bin/env bash
# Run only on the selected AutoDL research instance, after rental authorization.
# This script never creates or pays for a cloud instance.
set -euo pipefail
cd "$(dirname "$0")/.."
run_root="${CAEA_GPU_RUN_ROOT:-/root/autodl-tmp/caea-bf16}"
mkdir -p "$run_root"
if [[ "${CAEA_PROVIDER_TIMER_CONFIRMED:-0}" != "1" ]]; then
  echo 'Set and verify the provider shutdown timer first; then set CAEA_PROVIDER_TIMER_CONFIRMED=1.' >&2
  exit 2
fi
stage="${CAEA_GPU_STAGE:-smoke}"
if [[ "$stage" != "smoke" && "$stage" != "full" ]]; then
  echo 'CAEA_GPU_STAGE must be smoke or full' >&2
  exit 2
fi
finish() {
  status=$?
  set +e
  tar -czf "$run_root/CAEA_GPU_results_partial_or_complete.tar.gz" -C "$run_root" --exclude='weights' --exclude='*.tar.gz' .
  if [[ "${CAEA_AUTODL_SHUTDOWN:-0}" == "1" ]]; then shutdown -h now; fi
  exit "$status"
}
trap finish EXIT
exec > >(tee -a "$run_root/gpu.log") 2>&1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
infer_options=()
minimum_free_gb=35
if [[ "${CAEA_LOCAL_WEIGHTS:-0}" == "1" ]]; then
  infer_options+=(--local-weights)
  minimum_free_gb=5
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
nvidia-smi
if [[ "${CAEA_INSTALL_GPU_DEPS:-1}" == "1" ]]; then
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-gpu.txt
fi
python code/gpu_study.py preflight --out "$run_root/results" --weights "$run_root/weights" --minimum-free-gb "$minimum_free_gb"
if [[ "$stage" == "smoke" ]]; then
  python code/gpu_study.py prepare --out "$run_root/smoke"
  timeout 30m python code/gpu_study.py infer --model qwen15 --limit 8 --out "$run_root/smoke" --weights "$run_root/weights" "${infer_options[@]}"
  echo 'Smoke scoring finished. This is not a completed study. Inspect latency, memory and output before a full run.'
  exit 0
fi
python code/gpu_study.py prepare --out "$run_root/results"
for model in qwen15 smol17 qwen7; do
  timeout 90m python code/gpu_study.py infer --model "$model" --out "$run_root/results" --weights "$run_root/weights" "${infer_options[@]}"
done
python code/gpu_study.py verify --out "$run_root/results"
python code/gpu_study.py analyse --out "$run_root/results"
