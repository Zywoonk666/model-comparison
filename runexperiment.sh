#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_name="experiment-v2-progressive-mask"
output_root="${OUTPUT_ROOT:-$repo_dir/cache/experiments/$experiment_name}"
model="all"
launcher_args=()

usage() {
    cat <<'HELP'
用法：bash runexperiment.sh [--model MODEL] [--output_root DIR] [比较入口参数]

默认依次运行 GIFT、GIMLET-original、GIMLET-adapter；失败时停止。
  --model MODEL       all（默认）、gift、gimlet-original、gimlet-adapter
  --output_root DIR   总结果目录；每个模型使用独立子目录
                     相对路径从本脚本所在的仓库根目录计算
  --dry_run           仅检查清单及启动命令，不加载模型或开始训练
  --help              显示本说明

其他参数原样传给比较入口，例如 --datasets bace bbbp、--device 0、
--data_root /path/to/data。PYTHON、DEVICE、DATASETS、HF_OFFLINE 等
原有环境变量仍然有效；OUTPUT_ROOT 可指定总结果目录。
同一结果目录不能重复运行；重跑请指定新的 --output_root。

示例：
  bash runexperiment.sh --dry_run
  bash runexperiment.sh --model gift --datasets bace
  bash runexperiment.sh --output_root cache/experiments/repeat_2
HELP
}

while (($#)); do
    case "$1" in
        --model|--output_root)
            if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
                printf '错误：%s 需要一个值。\n' "$1" >&2
                exit 2
            fi
            if [[ "$1" == --model ]]; then model="$2"; else output_root="$2"; fi
            shift 2
            ;;
        --model=*) model="${1#*=}"; shift ;;
        --output_root=*) output_root="${1#*=}"; shift ;;
        --output_dir|--output_dir=*)
            printf '错误：统一入口请使用 --output_root；脚本会为各模型分配独立目录。\n' >&2
            exit 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) launcher_args+=("$1"); shift ;;
    esac
done

case "$model" in
    all) models=(gift gimlet-original gimlet-adapter) ;;
    gift|gimlet-original|gimlet-adapter) models=("$model") ;;
    *) printf '错误：不支持的模型 %s；使用 --help 查看选项。\n' "$model" >&2; exit 2 ;;
esac
if [[ -z "$output_root" ]]; then
    printf '错误：--output_root 不能为空。\n' >&2
    exit 2
fi
if [[ "$output_root" != /* ]]; then output_root="$repo_dir/$output_root"; fi

for model in "${models[@]}"; do
    case "$model" in
        gift) entry="$repo_dir/GIFT/run_gift_comparison.sh" ;;
        gimlet-original) entry="$repo_dir/GIMLET-original/run_experiments.sh" ;;
        gimlet-adapter) entry="$repo_dir/GIMLET-adapter/run_experiments.sh" ;;
    esac
    printf '\n运行 %s（%s）；结果目录：%s\n' "$model" "$experiment_name" "$output_root/$model"
    bash "$entry" "${launcher_args[@]}" --output_dir "$output_root/$model"
done
