# 统一实验入口

两个实验分支的仓库根目录均提供 `runexperiment.sh`。先激活已安装 GIFT/GIMLET
依赖的训练环境，再运行脚本。默认依次执行 GIFT、GIMLET-original、GIMLET-adapter，
一个模型完成后才启动下一个；任一模型失败即停止。

```bash
cd /root/model-comparison
conda activate env_gift

# 检查当前分支三个模型的启动命令和共享清单，不训练。
bash runexperiment.sh --dry_run

# 按当前分支的模型实现，运行三个模型的完整比较。
bash runexperiment.sh

# 只训练当前分支的 GIFT；已跑过基线时可用此方式。
bash runexperiment.sh --model gift
```

`--model` 可选 `all`、`gift`、`gimlet-original`、`gimlet-adapter`。
脚本使用已有比较入口，共享原来的数据划分、few-shot 样本清单及训练/评估设置。
不包含 Graph2Token。完整训练仍需准备预训练权重、数据及兼容环境。
默认 Hugging Face 离线；需要联网加载权重时可设置 `HF_OFFLINE=0`。

## 分支和结果目录

| 分支 | GIFT 版本 | 默认总结果目录（相对仓库根目录） |
| --- | --- | --- |
| `experiment-v2` | OT 排除 PAD，训练 mask 固定 | `cache/experiments/experiment-v2/` |
| `experiment-v2-progressive-mask` | OT 排除 PAD，训练 mask 按 epoch 递减 | `cache/experiments/experiment-v2-progressive-mask/` |

每个总目录下分别创建 `gift/`、`gimlet-original/`、`gimlet-adapter/` 子目录，
各自保存每个数据集的 CSV、日志和 `run_config.json`。
运行期间不要切换同一工作区的分支；切换分支不会删除忽略的结果文件。

```bash
# 确认没有训练正在运行后，切换到需要的版本。
git switch experiment-v2
bash runexperiment.sh --dry_run

# 或者使用包含按 epoch 渐进 mask 的版本。
git switch experiment-v2-progressive-mask
bash runexperiment.sh --dry_run
```

重复实验、改变数据集范围或超参数时，使用新的总结果目录。已有比较入口会拒绝
写入非空模型结果目录，不会自动覆盖、跳过或续跑。
`--output_root` 的相对路径始终从仓库根目录计算，也支持绝对路径。
统一入口使用 `--output_root`，不要传单模型入口的 `--output_dir`。

```bash
bash runexperiment.sh --output_root cache/experiments/repeat_2

# 单数据集、指定设备。
bash runexperiment.sh --model gift --datasets bace --device 0 \
  --output_root cache/experiments/gift_bace_trial

# 指定 Python 和数据缓存；对三个模型生效。
PYTHON=/path/to/env/bin/python DATA_ROOT=/path/to/compatible/data \
  OUTPUT_ROOT=/path/to/new_results bash runexperiment.sh
```

其他参数原样传给已有比较入口；`DEVICE`、`DATASETS`、`RUNSEED`、`FEW_SHOT`、
`PROTOCOL_DIR`、`MODEL_NAME_OR_PATH`、`TOKENIZER_NAME`、`ADAPTER_DIM` 等原有
环境变量继续有效。更改 seed/few-shot 前，需要准备对应的共享样本清单。

## 渐进 mask 分支的参数

以下参数只在 `experiment-v2-progressive-mask` 分支支持，实际仅改变 GIFT：

```bash
bash runexperiment.sh --model gift --mask_start_ratio 1 --mask_end_ratio 0.2 \
  --output_root cache/experiments/gift_mask_end02
```

默认训练 mask 为 `1 -> 0`；验证/测试仍采用原有确定性分层 mask。
`MASK_START_RATIO`、`MASK_END_RATIO` 环境变量同样可用。
模型改动与验证范围见 [GIFT/README_MODEL_VARIANTS.md](GIFT/README_MODEL_VARIANTS.md)。

## 推送两个分支

本地改动提交后，在可完成 GitHub 身份验证的终端执行：

```bash
cd /root/model-comparison
git push --atomic -u origin experiment-v2 experiment-v2-progressive-mask
```

该命令同时推送两个分支，与当前检出的分支无关。
