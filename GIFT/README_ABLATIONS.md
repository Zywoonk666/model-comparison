# GIFT 三模块消融（4090）

只跑 BACE、HIV、BBBP、Tox21、ESOL、FreeSolv 分子属性预测，沿用当前
`experiment-v2-progressive-mask` 的模型代码，不切换工作树，不增加生成任务或训练目标。

## 实验矩阵

| 输出子目录 | OT 对齐 | 渐进 mask | 图专家 | 用途 |
|---|---|---|---|---|
| `full` | 开 | 逐层放开 + epoch 比例 1→0 | 开 | 完整 GIFT；复用已完成、经过核验的结果 |
| `no_ot` | 关 | 与 full 相同 | 开 | OT 的条件贡献 |
| `no_progressive_mask` | 开 | 关：所有层固定屏蔽图 query→文本 key | 开 | 整个渐进 mask 模块的条件贡献 |
| `no_graph_expert` | 开 | 与 full 相同 | 关：保留原始预训练 T5 block | 图专家模块的条件贡献 |
| `fixed_epoch_mask` | 开 | 保留逐层放开，epoch 比例固定为 1 | 开 | 仅检验随 epoch 放开的贡献 |

完整模型在前 `floor(0.8 * 层数)` 层应用 mask，后面的层允许图 query 关注有效文本 key。
训练时 mask 概率从 1 线性降到 0；评估时前述早期层使用确定性屏蔽。
`no_progressive_mask` 同时取消逐层和逐 epoch 放开，因此额外增加 `fixed_epoch_mask`
用于拆分这两个因素。后者的评估策略与 full 相同。
`no_progressive_mask` 的命令保留默认 `mask_start_ratio=1, mask_end_ratio=0`，但在关闭开关后这两个值不参与计算。
训练 CSV 记录实际生效的固定比例 `1→1`；汇总校验也按实际生效值判断。

所有组保留相同 projector：2 个残差块、ReLU、dropout 0.5。只训练原实验规定的
lm_head、projector 和存在时的图专家；去掉图专家后不会解冻整个 T5 代替它。
因此 `no_graph_expert` 也改变可训练参数量，这是模块移除实验，不是参数量匹配实验。
OT 是现有编码器中的传输与融合步骤，不添加辅助 loss。

## 固定的比较条件

- 环境：`/root/anaconda3/envs/env_gift`，离线预训练模型 `haitengzhao/gimlet`，tokenizer `t5-small`。
- 数据：`/root/model-comparison/cache/property_data`。
- 共享切分：`/root/model-comparison/GIFT/cache/graph2token_protocol_4090` 中 17 个 manifest。
  不使用旧的 `graph2token_protocol` 目录。
- scaffold 80/10/10；few_shot=128（按原代码每标签分组的采样上限，不是总样本数）；runseed=6，seed=42。
- 60 epochs；effective batch=128、micro batch=16、梯度累积=8。
- Adam；lr=0.01、weight_decay=0；图专家 lr=0.0001。
- 按原实现第 19/39/59 epoch 评估，按验证集选择对应的测试分数。
- 分类为 ROC-AUC，越高越好；回归沿用 residual_std_ddof1，越低越好，不能称为 RMSE。
- Tox21 先按 property 汇总，12 个 property 等权算术平均；未跑齐 12 项时不填汇总分数。
- 本轮为配对单种子消融。可比较当前设置下的变化，不能声称统计显著或估计三个模块的完整交互效应。

## 执行

在仓库根目录执行，输出目录必须是不存在的新路径。脚本检测 GPU 是否空闲，并用锁阻止重复队列。
每次只运行一个训练进程，失败立即停止并记录原因。已启动的队列不要重复执行以下训练命令。

```bash
bash run_gift_ablations_4090.sh \
  --output_root cache/ablations/gift_NEW_RUN \
  --reuse_full cache/experiments/4090_20261114_132228/gift
```

加 `--dry_run` 只核对命令、manifest、完整模型结果、源代码和环境，不使用 CUDA。
`--reuse_full` 只允许原命令（除输出路径外）、所有 protocol hash、运行时源码和环境一致的已完成参考作业。
参考 CSV 复制到 `full/`，其来源和 SHA256 存入 `run_config.json`，不冒充重新训练。

对已停止且标记 `failed` 的队列，可以在相同命令后加 `--resume` 恢复。
恢复会重新核验配置、代码、环境、参考结果及已完成 CSV，保留并跳过完整的训练结果，
在原目录继续后续任务。旧失败状态和恢复脚本快照存入 `recovery/`，总日志追加保存。
存在未完成的 CSV 或仅有部分训练日志时，脚本拒绝自动覆盖，需要单独处理该任务。
恢复正在运行的队列会被状态检查和 GPU 锁拒绝。

新训练按数据集顺序执行：BACE → HIV → BBBP → Tox21 → ESOL → FreeSolv；
每个数据集依次运行 no_ot → no_progressive_mask → no_graph_expert → fixed_epoch_mask。
共新增 24 个数据集进程、68 个 property 训练；full 的 17 个 property 复用。

## 输出和查看

`cache/latest_gift_ablation.txt` 保存最新正式消融根路径。根目录文件：

- `RESULTS.md`：直接查看的六数据集、五组对照表，未完成标明“待完成”，并显示队列状态及失败原因。
- `summary.csv`：同一张表的完整精度数值；每个数据集进程结束后自动更新。
- `results_long.csv`：逐任务/逐 prompt 的测试分数、验证分数、best_epoch、protocol hash。
- `deltas.csv`：消融组相对 full 的差；`degradation_vs_full > 0` 表示去掉模块后变差。
- `status.json`：队列状态、当前数据集、配置、训练 PID 和已完成列表。
- `progress.json`：各组结果行数及是否完整。
- `training.log`：全部新训练日志；`pid` 是队列管理进程 PID。
- `run_config.json`、`environment.freeze.txt`、`run_ablations.snapshot.py`：配置、来源、环境和调度器快照。
- `exit_code`、`finished_at`：队列结束时写入；全部成功 `exit_code=0`。

每个组单独一个子目录，含 `run_config.json`、六个 `<dataset>.csv`；新训练组还有六个 `<dataset>.log`。
复用的 `full` 的原始日志仍保存在原三模型实验目录。与原实验一致，不额外保存模型 checkpoint。

```bash
ablation_root="$(cat cache/latest_gift_ablation.txt)"
cat "$ablation_root/status.json"
cat "$ablation_root/RESULTS.md"
tail -n 30 "$ablation_root/training.log"
nvidia-smi
```

需要在 Tox21 训练中读取已完成 property 时，`<组名>/tox21.csv` 已保留逐项结果。
汇总器也可单独执行；它不会启动训练：

```bash
bash run_gift_ablations_4090.sh --output_root "$ablation_root" --summarize_only
```

## 验证

`tests/test_ablations.py` 核对每组只改变预定参数、拒绝协议不匹配和重复结果、
以及只有 12 个 Tox21 任务齐全才计算均值。原切分、mask、OT 测试一起运行。
`verify_ablation_gpu.py` 使用真实离线 checkpoint 和 BACE 分子，在 CUDA 上验证五组的
训练前向、反向、正确的优化器参数组和评估前向，并检查 OT 实际调用次数、
去掉图专家后的 T5 block、梯度有限性和可训练参数数量。验证结果与正式作业分别保存。
