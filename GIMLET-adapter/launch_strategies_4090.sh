#!/usr/bin/env bash
# Detach the supervisor and all training children from SSH / VS Code terminals.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(dirname "$repo_dir")"
export PYTHON="${PYTHON:-/root/anaconda3/envs/env_gift/bin/python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export TZ=Asia/Shanghai

output_root=""
foreground=false
for ((i=1; i<=$#; i++)); do
    arg="${!i}"
    case "$arg" in
        --output_root)
            next=$((i+1)); output_root="${!next}" ;;
        --output_root=*) output_root="${arg#*=}" ;;
        --dry_run|--summarize_only|--help|-h) foreground=true ;;
    esac
done
if [[ -z "$output_root" ]]; then
    output_root="$workspace_dir/cache/adapter_strategies/strategies_$(date +%Y%m%d_%H%M%S)"
    set -- --output_root "$output_root" "$@"
fi
if [[ "$foreground" == true ]]; then
    exec "$PYTHON" "$repo_dir/run_adapter_strategies.py" "$@"
fi
output_root="$(realpath -m "$output_root")"
mkdir -p "$(dirname "$output_root")"
stamp="$(date +%Y%m%d_%H%M%S)"
launcher_log="${output_root}.${stamp}.launcher.log"
nohup setsid "$PYTHON" "$repo_dir/run_adapter_strategies.py" "$@" \
    > "$launcher_log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "${output_root}.launcher.pid"
printf '后台队列 PID: %s\n输出目录: %s\n启动日志: %s\n' "$launcher_pid" "$output_root" "$launcher_log"
