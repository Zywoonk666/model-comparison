# GIFT

This is the codebase for **GIFT**, which is built and improved upon the GIMLET framework for molecule learning. GIFT is a unified graph-text model tailored and evaluated focusing deeply on **Few-Shot Learning** for downstream tasks.

## Installation

The fixed environment targets Linux x86_64, Python 3.7, PyTorch 1.9, and
CUDA 11.1. Create it with:

### Environment
```bash
conda env create -f environment.yml
conda activate env_gift
```

### Pre-download Hugging Face Assets

`run_experiments.sh` enables Hugging Face offline mode. On a clean machine,
download the required model and tokenizer once while network access is
available:

```bash
conda activate env_gift

python - <<'PY'
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

snapshot_download(
    repo_id="haitengzhao/gimlet",
    allow_patterns=["config.json", "generation_config.json", "pytorch_model.bin"],
)
AutoTokenizer.from_pretrained("t5-small")
print("Hugging Face assets are ready for offline use.")
PY
```

By default, these files are stored in `~/.cache/huggingface/`. The download
must be performed for the same user that will run the experiment. After the
cache is populated, no network connection is required by the experiment
script. If the target machine has no network access, copy the populated
Hugging Face cache from another machine before running.

## Dataset Download

The MoleculeNet datasets (including pcba, bace, hiv, muv, tox21, toxcast, bbbp, esol, lipo, and freesolv) can be downloaded manually or automatically upon the first run. 
To download manually:
```bash
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
mv dataset property_data
```

## Run Few-Shot Learning Experiments

We provide an integrated script to run the few-shot learning pipeline, which evaluates the model natively on downstream tasks.

To execute the reproduction of the few-shot learning downstream task, run:
```bash
bash run_experiments.sh
```

The script automates few-shot testing (e.g. on `bace`) leveraging the GIFT custom architecture. All evaluation results will be cached as CSV files in the `./cache/` directory.

## Graph2Token comparison

Graph2Token is cloned as a separate sibling repository at `../Graph2Token`.
See [its comparison guide](../Graph2Token/README_GIFT.md) for the aligned protocol,
checkpoint requirements, and the legacy sampling/metric details.
Export the actual molecules selected during a GIFT run with:

```bash
EXPORT_FEW_SHOT_DIR=cache/graph2token_protocol bash run_experiments.sh
```

Alternatively, `python export_graph2token_data.py` exports a fresh default run's
data without loading GIFT weights. Graph2Token reads these manifests directly,
so it does not independently resplit or resample the molecules.

## Four-model baseline scripts

See [README_BASELINES.md](README_BASELINES.md) for the GIFT, GIMLET-original,
GIMLET-adapter, and Graph2Token comparison. For exact shared-manifest runs use:

```bash
bash run_gift_comparison.sh --dry_run
bash run_gift_comparison.sh
```

Each other repository now has `run_experiments.sh`. New comparison results carry
sample/protocol hashes and can be checked and combined with `summarize_baselines.py`.
The original `run_experiments.sh` remains available with its existing behavior.
