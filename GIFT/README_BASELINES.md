# 四个模型的基线运行说明

这四个目录分别是 GIFT、GIMLET-original、GIMLET-adapter 和 Graph2Token。
GIFT 原有脚本的准确名称为 `run_experiments.sh`（复数）。
另外三个仓库现在也提供同名入口；正式四模型比较建议让 GIFT 使用新入口
`run_gift_comparison.sh`，四者读取相同的样本清单并在结果中记录协议摘要。

| 仓库 | 推荐比较入口 | 更新的参数 |
| --- | --- | --- |
| GIFT | `bash run_gift_comparison.sh` | `innovation`：head、projector、graph expert |
| GIMLET-original | `bash run_experiments.sh` | `linear`：lm_head，包含 T5 共享嵌入权重 |
| GIMLET-adapter | `bash run_experiments.sh` | `adapter`：各层 Adapter 与 lm_head，默认瓶颈 64 |
| Graph2Token | `bash run_experiments.sh` | 原 Graph2Token 可训练模块，LLM 与 mapping 冻结 |

GIMLET 两个仓库原有 `downstream_test.sh` 默认 runseed=5、包含更多数据集，
也不会自动选择 Adapter 模式，因此不作为这次比较入口。
Graph2Token 的新同名脚本调用已有 `run_gift_comparison.sh`。
三个 GIMLET 系列入口共用本目录的 `run_gimlet_baseline.py` 和
`comparison_protocol.py`，请保持四个仓库并列；复制到其他机器时保留这个关系。

## 一次准备共同数据

本机已经生成默认 `128 / seed 6` 的 6 个数据集、17 个任务清单，存放在
`GIFT/cache/graph2token_protocol/`。清单含分子 ID、SMILES、原始标签、划分、
提示词和训练参数。它对应一个全新默认 GIFT 运行，不是旧结果的追溯证明。

新机器或更改 few-shot/seed 时先导出一次：

```bash
cd /root/GIFT
# 使用装有 torch、PyG、RDKit、OGB、pandas、sklearn 的环境。
# 当前机器可用 /root/Graph2Token/.venv-test/bin/python 进行数据导出。
python export_graph2token_data.py --few_shot 128 --runseed 6
```

也可从实际 GIFT 训练中导出：

```bash
EXPORT_FEW_SHOT_DIR=cache/graph2token_protocol bash run_experiments.sh
```

从已有结果跳过的任务不会导出。定制/恢复训练的数据以实际运行导出为准。
新协议不能覆盖同名但内容不同的清单，需要选择新 `--output_dir`。

GIFT/GIMLET 两个基线运行时读取各自 `property_data/`，没有数据会按原 loader 下载。
它们会核对清单中每个 ID 的 SMILES 和标签，之后直接使用清单中的三组索引，
**不再各自重新抽样**。如果本地数据顺序/标签不一致会报错。
可用 `DATA_ROOT` 指向与其 PyG 版本兼容的共同缓存，避免重复预处理。
不要把 Graph2Token/PyG 2 导出的 `.pt` 缓存直接作为旧 PyG 1 的缓存；
默认导出源数据单独存在 `GIFT/cache/graph2token_source_data/`。

## 环境与运行

GIFT 和两个 GIMLET 版本使用 GIFT README 描述的 `env_gift` 兼容环境。
先准备 `haitengzhao/gimlet` 权重和 `t5-small` tokenizer；三者默认使用该预训练来源。
脚本默认开启 Hugging Face 离线模式；需要联网加载时设 `HF_OFFLINE=0`。
Graph2Token 使用独立环境，安装方法及权重要求见
[Graph2Token/README_GIFT.md](../Graph2Token/README_GIFT.md)。

四个入口均支持不加载模型、不启动训练的检查，可用普通 Python 执行：

```bash
bash /root/GIFT/run_gift_comparison.sh --dry_run
bash /root/GIMLET-original/run_experiments.sh --dry_run
bash /root/GIMLET-adapter/run_experiments.sh --dry_run
bash /root/Graph2Token/run_experiments.sh --dry_run
```

正式训练在对应环境分别执行：

```bash
conda activate env_gift
bash /root/GIFT/run_gift_comparison.sh
bash /root/GIMLET-original/run_experiments.sh
bash /root/GIMLET-adapter/run_experiments.sh
```

Graph2Token 需先设置实际权重路径：

```bash
# 激活 Graph2Token 自己的环境后：
export LLM_MODEL=/path/to/Llama3-8B
export GNN_CHECKPOINT=/path/to/molecule_model_final.pth
export MOL2LATENT_CHECKPOINT=/path/to/mol2latent_model_final.pth
export STAGE2_CLS=/path/to/classification_stage2.ckpt
export STAGE2_REG=/path/to/regression_stage2.ckpt
# 若删减版 Stage 2 缺少词表映射层，还需 MAPPING_CLS / MAPPING_REG。
bash /root/Graph2Token/run_experiments.sh
```

共同的环境变量（示例）：

```bash
DATASETS="bace bbbp" DEVICE=0 RUNSEED=6 FEW_SHOT=128 \
  bash /root/GIMLET-original/run_experiments.sh

PYTHON=/path/to/env/bin/python OUTPUT_DIR=/path/to/new_results \
  bash /root/GIMLET-adapter/run_experiments.sh
```

`PYTHON` 选择解释器，`DEVICE` 设置 CUDA_VISIBLE_DEVICES，`PROTOCOL_DIR` 选择清单目录，
`OUTPUT_DIR` 选择结果目录。`RUNSEED/FEW_SHOT` 选择对应清单，不会自行改变其中的实验条件。
训练轮数、lr、batch、累积步数、评估频率都从清单读取；变更时先导出新清单。
T5 三个入口也支持 `--datasets bace bbbp`、`--output_dir`、`--data_root` 等参数，
GIMLET-adapter 可用 `ADAPTER_DIM=64`；Graph2Token 的模型参数见其 `--help`。

## 默认比较条件和结果位置

共同默认：bace/hiv/bbbp/tox21/esol/freesolv；scaffold 80/10/10；runseed=6；
few_shot=128；Adam、lr=0.01、weight_decay=0；60 轮；有效 batch=128，
micro batch=16，累积 8 次；按 GIFT 实际代码在第 19/39/59 轮评估。
每个属性独立训练。选择验证集最佳轮次对应的测试成绩；验证指标无效时明确报错，
不会依据测试集选模型。默认不从旧 CSV 跳过任务或自动断点续跑。

| 模型 | 默认结果目录（相对各自仓库） |
| --- | --- |
| GIFT | `cache/baselines/gift/shot128_seed6/` |
| GIMLET-original | `cache/baselines/gimlet_original/shot128_seed6/` |
| GIMLET-adapter | `cache/baselines/gimlet_adapter/shot128_seed6/` |
| Graph2Token | `cache/baselines/graph2token/shot128_seed6/` |

T5 三者每个数据集一个 CSV 和日志，并保存 `run_config.json`；Graph2Token 保存
`results.csv` 及每个任务的 JSON。共同关键列为 `dataset / split / prompt_id / runseed /
protocol_sha256 / metric / valid_best / test_best / best_epoch / n_train_labeled`。
读取 `test_best` 比较；`best_val_idx` 是评估列表索引，真实轮次看 `best_epoch`。
重复实验请使用新的输出目录。原 GIFT 的 `run_experiments.sh` 仍保留旧输出路径。

四个模型都跑完后生成对比表：

```bash
cd /root/GIFT
python summarize_baselines.py \
  --gift cache/baselines/gift/shot128_seed6 \
  --gimlet ../GIMLET-original/cache/baselines/gimlet_original/shot128_seed6 \
  --adapter ../GIMLET-adapter/cache/baselines/gimlet_adapter/shot128_seed6 \
  --graph2token ../Graph2Token/cache/baselines/graph2token/shot128_seed6
```

输出 `cache/baseline_comparison.csv`，按数据集和 seed 一行。分类 ROC-AUC 越大越好，
回归 residual_std_ddof1 越小越好。先在属性内平均提示，再对属性取等权均值，
所以 Tox21 是 12 个任务的宏平均；不会把分类和回归混在一起取平均。
汇总前检查四者的任务集合、协议摘要、指标、轮数、学习率，缺失或不匹配会报错。
旧 CSV 没有协议元数据，不能直接冒充这次严格对齐的新结果。

## 比较的边界

保留当前 GIFT 的抽样定义：128 是每个 `int(label)` 分组上限；二分类通常 256 条，
ESOL 当前 774 条，FreeSolv 513 条。Tox21 原代码逐任务缩小抽样池，且缺失标签可能
占训练 ID；实际有效标签数记录在结果中。这不是统一“总计 128 条”的协议。
GIFT 原名为 RMSE 的指标实为残差样本标准差，本对比明确标注该名称；Graph2Token
还在 JSON 中报告标准 RMSE/MAE。T5 的数值文本生成与 Graph2Token 的数值头仍有
预测格式和有效预测覆盖率差异。各模型架构、预训练规模和可训练参数量也不同。

Graph2Token 默认 `graph_without_iupac`，属于输入消融；启用 IUPAC 时须另标输入条件。
Stage 2 是否用过目标任务额外标签需核实。统一学习率提供相同固定配置，
不等同于各模型都完成了最佳超参数搜索。单个 seed 结果也不代表多次运行的均值/方差。

本次检查覆盖脚本 dry-run、协议导入、样本/标签检查、正确训练模式和结果汇总。
没有完成四个预训练模型的完整训练；当前机器缺少可用 GPU 驱动和所需预训练权重。
