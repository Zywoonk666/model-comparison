#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"
conda_base="$(conda info --base)"
export PYTHON="${PYTHON:-$conda_base/envs/env_gift/bin/python}"
export CUDA_VISIBLE_DEVICES="${DEVICE:-0}"
export DEVICE="${DEVICE:-0}"
export DATA_ROOT="${DATA_ROOT:-$repo_dir/cache/property_data}"
export PROTOCOL_DIR="${PROTOCOL_DIR:-$repo_dir/GIFT/cache/graph2token_protocol_4090}"
export HF_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is unavailable in env_gift'
print('Python environment:', __import__('sys').executable, flush=True)
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, flush=True)
print('GPU:', torch.cuda.get_device_name(0), flush=True)
x = torch.randn(128, 128, device='cuda', requires_grad=True)
x.square().mean().backward()
torch.cuda.synchronize()
print('CUDA forward/backward: PASS', flush=True)
PY

exec bash "$repo_dir/runexperiment.sh" "$@"
