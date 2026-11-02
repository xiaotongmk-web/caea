#!/usr/bin/env bash
# Run only on the selected AutoDL research instance, after rental authorization.
# This script never creates or pays for a cloud instance.
set -euo pipefail
cd "$(dirname "$0")/.."
run_root="${CAEA_GPU_RUN_ROOT:-/root/autodl-tmp/caea-bf16}"
mkdir -p "$run_root"
finish() {
  status=$?
  set +e
  tar -czf "$run_root/CAEA_GPU_results_partial_or_complete.tar.gz" -C "$run_root" results gpu.log
  if [[ "${CAEA_AUTODL_SHUTDOWN:-0}" == "1" ]]; then shutdown -h now; fi
  exit "$status"
}
trap finish EXIT
exec > >(tee -a "$run_root/gpu.log") 2>&1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
nvidia-smi
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-gpu.txt
python code/gpu_study.py prepare --out "$run_root/results"
for model in qwen15 smol17 qwen7; do
  timeout 90m python code/gpu_study.py infer --model "$model" --out "$run_root/results" --weights "$run_root/weights"
done
python code/gpu_study.py analyse --out "$run_root/results"
