#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"
exec "${PYTHON:-python}" "$repo_dir/../GIFT/run_gimlet_baseline.py" \
    --repo "$repo_dir" --variant gift "$@"
