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
