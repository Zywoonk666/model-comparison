#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"
conda_base="$(conda info --base)"
export PYTHON="${PYTHON:-$conda_base/envs/env_gift/bin/python}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec "$PYTHON" "$repo_dir/GIFT/run_ablations.py" "$@"
