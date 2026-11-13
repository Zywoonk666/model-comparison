# GIFT 模型改动分支

本文件描述 `/root/model-comparison` 合并仓库中的 GIFT 变体。
分支切换作用于整个合并仓库；训练时使用其中的 `GIFT/run_gift_comparison.sh`。

## experiment-v2：排除 OT 中的 PAD

OT 接收文本 `attention_mask` 和图编码器返回的有效位置标记。
文本每个有效 token 分配 `1 / 有效文本长度` 的边缘质量；图侧每个有效位置
分配 `1 / 有效图长度`。虚拟图汇总 token 仍为有效位置，PAD 的边缘质量及
对应运输矩阵行/列均为零。全 PAD 的任意一侧会报错。

该分支保留原来的层间 mask：前 `floor(层数 * 0.8)` 层屏蔽图读取文本，
后面的层放开；训练 epoch 间的 mask 比例仍固定为 1。

在已安装 GIFT 兼容依赖的环境中运行（测试使用 Transformers 4.28.1）：

```bash
cd /root/model-comparison/GIFT
python -m pytest -q tests/test_ot_alignment.py tests/test_comparison_launchers.py
```

测试覆盖 PAD 零质量、有效位置归一化、不同 padding 长度和同批其他样本不改变
OT 结果、有限梯度，以及小型真实 GIFT 模型的前向/反向传播。测试不需要下载权重
或数据；只替换与 GIFT 无关的其他基线导入，不替换 GIFT/T5/GraphEncoder 计算。

## experiment-v2-progressive-mask：OT 修复 + 按 epoch 渐进 mask

该分支从 `experiment-v2` 创建，包含上述 OT 修复。训练时，第 `e` 轮的屏蔽概率为：

```text
p(e) = start + (end - start) * (e - 1) / (E - 1), e = 1, ..., E
```

默认 `start=1.0, end=0.0`：60 轮训练时，第 1 轮为 1，第 30 轮约 0.5085，
第 60 轮为 0。只有一轮时取 start；起止值必须满足 `0 <= end <= start <= 1`。
每个属性/提示词重新从第 1 轮开始调度。

前 `floor(层数 * 0.8)` 层中的“图 query 读取文本 key”连接按当前概率随机屏蔽；
后面的层保持放开。例如 6 层模型中，前 4 层按 epoch 调度，后 2 层始终放开。
比例是屏蔽概率，并非每批精确屏蔽相同比例；每次前向重新采样，因此同一条连接
可能在之后的某一批再次被屏蔽。图 PAD 和文本 PAD 的屏蔽始终保留。
`--use_progressive_mask False` 会关闭层间放开和 epoch 调度，所有层严格屏蔽该方向。

**验证/测试沿用原有确定性分层规则**：前 80% 层严格屏蔽该方向，后面的层放开，
不使用训练时的随机概率。因此这里实现的是训练阶段的渐进策略，训练后期和评估
的 attention mask 仍有差异。OT 本身仍可以注入有效文本信息。

使用新的结果目录区分两个分支（Git 切分支不会清理忽略的 cache 目录）：

```bash
cd /root/model-comparison

# OT 修复，保持固定训练 mask。
git switch experiment-v2
bash GIFT/run_gift_comparison.sh \
  --output_dir /root/model-comparison/GIFT/cache/baselines/gift_ot

# OT 修复，训练 mask 从 1 逐轮降至 0。
git switch experiment-v2-progressive-mask
bash GIFT/run_gift_comparison.sh \
  --output_dir /root/model-comparison/GIFT/cache/baselines/gift_ot_epoch
```

脚本会先进入 `GIFT`，相对 `--output_dir` 按该工作目录解析，因此上面使用绝对路径。
不要在训练运行期间切换同一工作区的分支。

可通过参数或环境变量指定起止概率：

```bash
bash GIFT/run_gift_comparison.sh --mask_start_ratio 1 --mask_end_ratio 0.2 \
  --output_dir /root/model-comparison/GIFT/cache/baselines/gift_ot_epoch_end02

MASK_START_RATIO=1 MASK_END_RATIO=1 bash GIFT/run_gift_comparison.sh \
  --output_dir /root/model-comparison/GIFT/cache/baselines/gift_ot_constant
```

`1 -> 1` 恢复 OT 修复分支的固定训练 mask。训练日志记录每轮实际概率，CSV 记录
调度名称、起止概率和评估概率；比较入口的 `run_config.json` 保存完整启动参数。
原始 `run_experiments.sh` 也使用新的 downstream 默认值；环境变量选项由
`run_gift_comparison.sh` 支持。GIMLET-original 和 GIMLET-adapter 的启动命令及共享
数据清单保持原样。原抽样、评估指标、验证选优和评估轮次均保持原样。

```bash
cd /root/model-comparison/GIFT
python -m pytest -q tests
```

额外验证涵盖调度端点、单轮训练、参数合法性、真实训练循环的参数传递，以及
真实 GIFT 层中屏蔽概率从全屏蔽到部分屏蔽、再到完全放开的行为和有限梯度。
这些检查不代替完整的 few-shot 训练，模型效果需要重新实验确定。
