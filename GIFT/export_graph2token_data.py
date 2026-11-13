"""Export a fresh default GIFT run without loading its pretrained model.

For resumed/skipped tasks or customized runs use --export_few_shot_dir on
downstream_test.py instead; that captures the actual runtime selection.
"""
import argparse
import importlib.util
import json
from pathlib import Path

from comparison_protocol import export_task, legacy_sample, settings_from_args


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["bace", "hiv", "bbbp", "tox21", "esol", "freesolv"])
    parser.add_argument("--data_root", type=Path, default=root / "cache/graph2token_source_data",
                        help="Separate cache avoids overwriting GIFT's older PyG serialization")
    parser.add_argument("--output_dir", type=Path, default=root / "cache/graph2token_protocol")
    parser.add_argument("--few_shot", type=int, default=128)
    parser.add_argument("--runseed", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--test_interval", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_accum_step", type=int, default=8)
    args = parser.parse_args()
    if min(args.few_shot, args.epochs, args.test_interval, args.batch_size, args.grad_accum_step) < 1:
        parser.error("counts must be positive")
    if args.few_shot < args.batch_size:
        args.grad_accum_step = int(args.grad_accum_step * args.few_shot / args.batch_size)
        args.batch_size = args.few_shot
    if args.grad_accum_step < 1 or args.batch_size % args.grad_accum_step:
        parser.error("This shot/batch combination is unsupported by the current GIFT script")
    if args.few_shot % args.batch_size:
        parser.error("GIFT requires few_shot to be divisible by effective batch_size")
    args.batch_size_ori = args.batch_size
    args.seed, args.split, args.decay = 42, "scaffold", 0.0
    settings = settings_from_args(args)
    if not settings["eval_epochs"]:
        parser.error("No evaluation epoch under GIFT's (epoch + 1) rule")
    import numpy as np
    import pandas as pd
    import torch
    # Import sklearn before OGB's background version-check thread starts;
    # older joblib/setuptools combinations can otherwise deadlock on imports.
    splitters = load_file("gift_splitters", root / "dataloaders/splitters.py")
    datasets = load_file("gift_molecule_datasets", root / "dataloaders/molecule_datasets.py")
    prompts = json.loads((root / "prompts/selected_prompt_downstream_task.json").read_text())
    for name in args.datasets:
        # Each dataset is a separate process in run_experiments.sh.
        rng = np.random.RandomState(args.runseed)
        dataset = datasets.MoleculeDatasetSplitLabel(root=str(args.data_root), name=name)
        smiles = pd.read_csv(args.data_root / name / "processed/smiles.csv", header=None)[0].tolist()
        train, valid, test = splitters.scaffold_split(torch.arange(len(smiles)), smiles)
        labels = dataset.data.y.numpy()
        for task in range(labels.shape[1]):
            if str(task) not in prompts[name]:
                continue
            # Preserve GIFT's cumulative support-set narrowing between tasks.
            train = legacy_sample(train, labels[:, task], args.few_shot, rng)
            path = export_task(args.output_dir, name, task, smiles, labels[:, task],
                               dict(train=train, valid=valid, test=test), prompts[name][str(task)],
                               settings, source="fresh_gift_default_export")
            print("{}: train={} valid={} test={}".format(path, len(train), len(valid), len(test)))


if __name__ == "__main__":
    main()
