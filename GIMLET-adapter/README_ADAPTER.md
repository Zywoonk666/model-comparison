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

## Four token adaptation strategies (4090 study)

The new study compares **shared**, **graph-only**, **text-only**, and **separate**
residual adapters, keeping the pretrained backbone, tied output/input embedding
training policy, data, prompts, optimizer, and effective batch size fixed.

| Strategy | Graph bottleneck | Text bottleneck | Adapter parameters (6 layers) | Total trainable |
|---|---:|---:|---:|---:|
| `shared` | 64, shared | 64, shared | 402,816 | 16,838,016 |
| `graph` | 64 | none | 402,816 | 16,838,016 |
| `text` | none | 64 | 402,816 | 16,838,016 |
| `separate` | 31 | 31, independent | 399,732 | 16,834,932 |

For hidden width d and bottleneck r, a bottleneck has `(2d+1)r+3d` parameters,
including LayerNorm and biases. Separate adapters have independent normalization,
down/up projections, and biases. Two width-32 adapters exceed the width-64 shared
budget. The runner therefore chooses the largest balanced integer widths under
the same budget ceiling: 31+31, **0.77% below** the shared count. This is a matched
ceiling within 1%, not mathematically identical counts or padding with unused
parameters. Both adapter-only and total trainable counts are recorded.

All four NEW groups use `--adapter_placement pre`: each adapter acts immediately
before its joint encoder block. This is necessary because GIMLET returns only
text states to the decoder: a graph-only adapter AFTER the final block has no
effect on prediction. Tests and real-checkpoint GPU validation check that the
last graph branch receives gradients. A graph token includes the virtual graph
token; the boundary uses the padded graph length, and existing attention masks
continue to ignore padded keys. Routing controls the direct residual update;
later attention may carry its effect to the other modality.

The default `shared/post` behavior remains compatible with existing checkpoints
and commands. The four-way runner explicitly uses `pre` for EVERY group and
re-trains shared. **Do not treat the old shared/post results as the new control.**
Non-shared/post configurations are rejected to prevent counting ineffective
last-layer graph parameters. No OT, progressive mask, or graph experts are added.

### Background execution, independent of VS Code / SSH

From `/root/model-comparison`:

```bash
# CPU-only validation: no training or CUDA allocation.
bash GIMLET-adapter/launch_strategies_4090.sh --dry_run

# nohup + setsid + detached stdin; four independent jobs, up to 40 GiB total budget.
bash GIMLET-adapter/launch_strategies_4090.sh \
  --output_root cache/adapter_strategies/my_new_run \
  --parallel 4 --memory_budget_gib 40
```

The launcher returns immediately and prints the PID, root, and external launcher
log path. GPU compute processes already running cause the supervisor to refuse
startup. Closing the terminal or disconnecting VS Code does not stop the queue;
server reboot/shutdown still stops processes. A GPU lock prevents duplicate
queues. A nonzero worker exit stops this queue and preserves its logs/results.
Offline runs also disable OGB 1.3.x's optional online version-check thread, which
otherwise may keep a finished process alive while waiting for an HTTP response.
The installed OGB version and all graph/data/model implementations are retained.

Parallelism uses the existing micro batch 16, gradient accumulation 8, effective
batch 128, and FP32; it does not silently change optimization or precision.
The 40 GiB budget reserves 2 GiB for CUDA contexts and non-allocator overhead,
then gives each of four workers a PyTorch allocator cap of 9.5 GiB. This is an
allocator/resource budget, not a promise to occupy all memory or a hardware-level
cap on non-PyTorch allocations. CPU threads are divided among workers (2 each
on the current 8-core machine). The runner reports actual peak allocator memory.

Six datasets only: BACE, HIV, BBBP, Tox21, ESOL, FreeSolv. Four jobs may run in
parallel; the pending list is dataset-first. All use the 17 existing manifests
at `GIFT/cache/graph2token_protocol_4090`, runseed 6, seed 42, 60 epochs, Adam
lr .01, and validation-selected evaluation at epochs 19/39/59. This is the legacy
sampling protocol: `few_shot=128` is not a uniform total of 128 labeled molecules.
Regression retains residual standard deviation (ddof=1), not RMSE. These choices
isolate strategy changes; this run does not establish a new strict few-shot or
publication-standard regression protocol.

### Outputs and recovery

`cache/latest_adapter_strategy_run.txt` points to the latest formal root:

```bash
strategy_root="$(cat cache/latest_adapter_strategy_run.txt)"
cat "$strategy_root/status.json"
cat "$strategy_root/RESULTS.md"
tail -n 30 "$strategy_root/graph/bace.log"
watch -n 2 nvidia-smi
```

- `RESULTS.md`, `summary.csv`: automatically updated after each complete dataset
  job. Tox21 is averaged only after all 12 properties finish; missing cells stay blank.
- `results_long.csv`: individual task scores and validation-selected epochs.
- `parameter_budget.csv`: exact parameter counts, widths, head policy, budget gap.
- `{shared,graph,text,separate}/{dataset}.{csv,log}`: raw results and training logs.
- `status.json`, `pid`, `exit_code`, `finished_at`: lifecycle, worker PIDs, errors.
- `run_config.json`: exact commands, manifest hashes, frozen source hashes, Git commit.
- `environment.freeze.txt`, supervisor snapshot: reproducibility information.

No checkpoints are saved by default, matching the previous benchmark. Resume a
stopped queue with the SAME arguments and `--resume`. Complete validated dataset
jobs are retained. Incomplete CSV/log files are archived under `recovery/`, and
that dataset's complete traversal starts over with its original seed. The queue
will not silently skip individual Tox21 tasks or mix partial runs. A changed
runtime source requires a fresh experiment directory.

```bash
bash GIMLET-adapter/launch_strategies_4090.sh \
  --output_root "$strategy_root" --parallel 4 --memory_budget_gib 40 --resume
```

Model tests cover zero-initialized identity, graph/text routing, parameter counts,
independent separate branches, gradients including the last graph branch,
checkpoint loading, and gradient checkpointing. Runner tests check shared
protocols, incomplete Tox21 handling, metadata rejection, and child scheduling.
`verify_strategy_gpu.py --output PATH` additionally exercises four real cached
GIMLET checkpoints on a batch of 16 actual BACE training molecules with CUDA.

### MUV, CYP450, and ToxCast extension

The extension adds all four pre-block strategies and the original shared/post
configuration (`--include_legacy_post`). It uses the same checkpoint, seed 6,
seed 42, scaffold 80/10/10, 60 epochs, Adam .01, FP32, micro batch 16 and
accumulation 8. The default launcher still selects only the original six datasets.

`prepare_extended_benchmarks.py` creates a NEW `independent_labeled_v1` protocol.
The legacy sampler cumulatively shrinks the training pool between tasks; an audit
found missing training classes in 15 MUV and 28 prompt-covered ToxCast tasks.
For the extension, each task samples up to 128 labeled molecules PER CLASS from
the full scaffold training pool, with NumPy seed `6 + task_id`. Missing labels
are excluded; validation/test keep all labeled molecules in their original
scaffold partition. This is explicitly a different sampling protocol from the
completed six-dataset study, whose results and manifests remain unchanged.

Task eligibility is frozen before training: a prompt and both classes in each
train/validation/test pool are required. The current data yield CYP450 5/5,
MUV 14/17 and ToxCast 573/617 eligible tasks. MUV excludes tasks 0 and 12 for
no test positives, and 16 for no validation positives. ToxCast excludes 31
missing prompts, 7 single-class validation tasks and 6 single-class test tasks.
`audit.json` records each included/excluded task, label column, class counts,
data/prompt hashes, and sampling seeds. These are reported as explicit subsets,
not full-task benchmark scores. A true zero AUC is allowed for this audited
protocol; absent-class tasks have already been excluded.

CYP450 comes from the ChemBench source specified in the original GIMLET README.
Its columns are CYP1A2, CYP2C19, CYP2C9, CYP2D6, CYP3A4. The original prompt file
swaps CYP2C19/CYP2C9; preparation corrects this ONLY in the new frozen prompt file.
MUV and ToxCast use the DeepChem downloads already specified by the data loader.

```bash
HF_HUB_OFFLINE=1 /root/anaconda3/envs/env_gift/bin/python \
  GIMLET-adapter/prepare_extended_benchmarks.py \
  --output_dir cache/adapter_protocols/extended_20261115

bash GIMLET-adapter/launch_strategies_4090.sh \
  --output_root cache/adapter_strategies/extended_20261115 \
  --datasets cyp450 muv toxcast --include_legacy_post \
  --protocol_dir /root/model-comparison/cache/adapter_protocols/extended_20261115 \
  --prompt_file /root/model-comparison/cache/adapter_protocols/extended_20261115/selected_prompt_downstream_task.json \
  --parallel 4 --memory_budget_gib 40
```

Use fresh paths for a new preparation or experiment. The extension is 15 dataset
jobs, 2,960 individual task runs, and can take much longer than the first six
datasets. The sequential order is CYP450, MUV, ToxCast, with four workers.
Dataset-level `--resume` preserves completed jobs but restarts an incomplete
dataset traversal; keep the background queue running for the long ToxCast jobs.

The extension also opts into sample-weighted gradient accumulation with an
optimizer update for the final incomplete window. Otherwise 97 eligible ToxCast
tasks with fewer than 128 training molecules would perform NO optimizer updates
in the legacy loop. Full windows still contain 128 molecules; the last window
uses its actual sample count and does not carry gradients across epochs. This
change applies only to `independent_labeled`, consistently across all five groups.

`update_benchmark_report.py --base_root OLD_RUN --extension_root NEW_RUN --watch`
merges the extension into the existing `OLD_RUN/RESULTS.md` and
`OLD_RUN/comparison_with_legacy.csv`. Run it with nohup/setsid as well. It shows
task completion counts while running, and publishes equal-task means only after
all eligible tasks pass supervisor validation. The old experiment's status and
raw results stay unchanged. `verify_extended_runtime.py` checks real CUDA
training and classification scoring for all five configurations plus the longest
eligible MUV/ToxCast prompts, using 16 large training molecules from each task.
