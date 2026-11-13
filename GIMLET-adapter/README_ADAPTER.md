# GIMLET + bottleneck Adapter

This worktree adds one residual bottleneck Adapter after each joint GIMLET
encoder block (self-attention and FFN), before graph/text states are split.
Within a layer, all graph and text tokens use the same Adapter. Each layer
has its own Adapter parameters.

```text
H_out = H + Up(ReLU(Down(LayerNorm(H))))
```

The default bottleneck width is 64. `Down.weight` uses Xavier uniform
initialization, and `Down.bias` is zero. Both `Up.weight` and `Up.bias` are
zero, so a new Adapter is exactly the identity function for finite inputs.
The original GIMLET weights are loaded from the pretrained checkpoint; no
additional pretraining stage is needed.

`--few_shot_fashion adapter` enables the adapters and trains their parameters
plus `lm_head`, following the output-head tuning choice used for the GIFT
comparison. Other parameters are frozen. T5's existing weight tying is
preserved: if `lm_head.weight` is shared with input embeddings, those embeddings
are updated too. The printed total trainable parameter count deduplicates
shared weights.

## Run the six-dataset GIFT comparison

Use `bash run_experiments.sh --dry_run` to check the shared manifests and
`bash run_experiments.sh` to train all six datasets in Adapter mode.
The default output is `cache/baselines/gimlet_adapter/shot128_seed6/`.
See [the four-model guide](../GIFT/README_BASELINES.md) for environment setup,
checkpoint preparation, comparison commands and result aggregation.

The manual example below retains the legacy independently sampled workflow.
Use the new script for exact molecule-level agreement with the other models.

## Run a few-shot experiment

Install the dependencies described in [README.md](README.md), and run from
this worktree. The following BACE example uses the existing GIFT experiment's
seed, learning rate, epochs, and batch settings:

```bash
cd /root/GIMLET-adapter
mkdir -p cache/gimlet_adapter_bace
python downstream_test.py \
  --transformer_backbone gimlet \
  --model_name_or_path haitengzhao/gimlet \
  --tokenizer_name t5-small \
  --dataset bace \
  --few_shot 128 \
  --few_shot_fashion adapter \
  --adapter_dim 64 \
  --runseed 6 \
  --batch_size 128 \
  --grad_accum_step 8 \
  --lr 1e-2 \
  --epochs 60 \
  --test_interval 20 \
  --transform_in_collator \
  --no_eval_train \
  --output_model_dir cache/gimlet_adapter_bace \
  --output_result_to_file cache/gimlet_adapter_bace.csv
```

The existing data sampling and evaluation routines are used. In this loader,
`--few_shot 128` samples up to 128 examples per label value, not 128 examples
in total. The original `downstream_test.sh` still selects its original training
modes; use the command above to select Adapter tuning explicitly.

The model state dictionary includes `encoder.adapters.*`. The model config
stores `adapter_dim`, so `save_pretrained()` / `from_pretrained()` on the
GIMLET model class can also save and restore the Adapter architecture and
weights. Loading the original checkpoint with Adapter tuning may report
missing `encoder.adapters.*` weights, since those modules are newly added.

## Focused checks

The standalone Adapter tests require PyTorch and pytest:

```bash
python -m pytest -q tests/test_adapter.py
```

With `transformers==4.28.1` installed, also run the tiny-model integration check:

```bash
python -m pytest -q tests
```

It checks checkpoint loading, identical initial logits/generation, the Adapter
input positions, gradient flow, frozen parameters, and saved-model restoration.
It uses synthetic inputs and a tiny local checkpoint. Only unused baseline
imports (GIN, KVPLM, MoMu, LLM) are replaced in the test; the GIMLET graph
embedding, T5 encoder/decoder, Adapter, and model-loading code are exercised.
This does not run a full MoleculeNet experiment.
