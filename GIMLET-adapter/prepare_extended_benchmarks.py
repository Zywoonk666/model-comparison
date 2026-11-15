"""Freeze auditable MUV/CYP450/ToxCast tasks and per-task labeled support sets.

Legacy cumulative support narrowing is not suitable for sparse multi-assay
datasets. The extension explicitly uses independent_labeled_v1; old manifests
and six-dataset scores are never overwritten.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

from offline_runtime import configure_offline_runtime
configure_offline_runtime()

REPO = Path(__file__).resolve().parent
WORKSPACE = REPO.parent
SOURCES = {
    'muv': ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz', 'muv.csv'),
    'toxcast': ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz', 'toxcast_data.csv'),
    'cyp450': ('https://raw.githubusercontent.com/shenwanxiang/ChemBench/master/src/chembench/data_and_index/CYP450/CYP450.csv.gz', 'CYP450.csv'),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_data(data_root, name):
    url, filename = SOURCES[name]
    root = data_root / name / 'raw'
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    if not path.exists():
        with urllib.request.urlopen(url, timeout=40) as response:
            compressed = response.read()
        content = gzip.decompress(compressed)
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(content)
        temporary.replace(path)
        (root / 'download.json').write_text(json.dumps(dict(url=url, sha256=digest(path)), indent=2) + '\n')
    provenance = json.loads((root / 'download.json').read_text())
    if digest(path) != provenance['sha256']:
        raise ValueError('Raw data checksum differs: ' + str(path))
    return path, provenance


def corrected_prompts(original, dataset, columns):
    prompts = original[dataset]
    if dataset != 'cyp450':
        return prompts, {int(k): int(k) for k in prompts}
    expected = ['label_1a2', 'label_2c19', 'label_2c9', 'label_2d6', 'label_3a4']
    if columns != expected:
        raise ValueError('Unexpected CYP450 column order: ' + repr(columns))
    mapping = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}
    result = {str(task): prompts[str(source)] for task, source in mapping.items()}
    for task, column in enumerate(columns):
        target = 'CYP' + column.split('_')[1].upper()
        if not all('particularly ' + target + ',' in text for text in result[str(task)]):
            raise ValueError('CYP450 prompt/target mismatch')
    return result, mapping


def labeled_support(pool, labels, shots, seed):
    import numpy as np
    rng = np.random.RandomState(seed)
    selected = []
    for label in (0., 1.):
        candidates = [int(i) for i in pool if labels[int(i)] == label]
        if not candidates:
            raise ValueError('Training pool must have both classes')
        selected.extend(rng.choice(candidates, min(shots, len(candidates)), replace=False).tolist())
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--data_root', type=Path, default=WORKSPACE / 'cache/property_data')
    parser.add_argument('--datasets', nargs='+', choices=list(SOURCES), default=['cyp450', 'muv', 'toxcast'])
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Use a fresh protocol directory')
    args.output_dir.mkdir(parents=True)
    import sys
    import numpy as np
    import pandas as pd
    import torch
    from dataloaders.molecule_datasets import MoleculeDatasetSplitLabel
    from dataloaders.splitters import scaffold_split
    sys.path.append(str(WORKSPACE / 'GIFT'))
    from comparison_protocol import export_task, load_task, settings_from_args

    source_prompt = REPO / 'prompts/selected_prompt_downstream_task.json'
    original = json.loads(source_prompt.read_text())
    selected = {}
    settings = settings_from_args(SimpleNamespace(few_shot=128, runseed=6, seed=42,
        split='scaffold', batch_size_ori=128, grad_accum_step=8, epochs=60,
        test_interval=20, lr=.01, decay=0., comparison_sampling_policy='independent_labeled'))
    audit = dict(protocol='independent_labeled_v1', source_prompt_sha256=digest(source_prompt),
                 sampling='Independent labeled training pool per task; max 128 per class; RNG seed=6+task',
                 selection='Prompt exists and train/valid/test each have both classes; no metric-based task selection',
                 settings=settings, datasets={})
    for name in args.datasets:
        raw, provenance = ensure_data(args.data_root, name)
        with raw.open() as stream:
            header = next(csv.reader(stream))
        columns = header[2:7] if name == 'cyp450' else header[:17] if name == 'muv' else header[1:618]
        prompts, mapping = corrected_prompts(original, name, columns)
        data = MoleculeDatasetSplitLabel(root=str(args.data_root), name=name)
        smiles = pd.read_csv(args.data_root / name / 'processed/smiles.csv', header=None)[0].tolist()
        labels = data.data.y.numpy()
        if labels.shape != (len(smiles), len(columns)):
            raise ValueError('Unexpected data shape')
        parts = scaffold_split(torch.arange(len(smiles)), smiles)
        base = {s: ids.tolist() for s, ids in zip(('train', 'valid', 'test'), parts)}
        (args.output_dir / (name + '_scaffold_indices.json')).write_text(json.dumps(base) + '\n')
        included, excluded = [], []
        selected[name] = {}
        for task, column in enumerate(columns):
            if str(task) not in prompts:
                excluded.append(dict(task=task, column=column, reasons=['missing_prompt']))
                continue
            y = labels[:, task]
            labeled = {s: [i for i in ids if np.isfinite(y[i]) and y[i] in (0., 1.)] for s, ids in base.items()}
            counts = {s: {str(label): sum(y[i] == label for i in ids) for label in (0, 1)} for s, ids in labeled.items()}
            reasons = [s + '_lacks_both_classes' for s in labeled if not all(counts[s].values())]
            if reasons:
                excluded.append(dict(task=task, column=column, reasons=reasons, counts=counts))
                continue
            labeled['train'] = labeled_support(base['train'], y, 128, 6 + task)
            path = export_task(args.output_dir, name, task, smiles, y, labeled, prompts[str(task)], settings,
                               source='adapter_extension_audited', protocol='independent_labeled_v1')
            manifest = load_task(path)
            selected[name][str(task)] = prompts[str(task)]
            included.append(dict(task=task, column=column, source_prompt_task=mapping[task], counts=counts,
                n_train=len(labeled['train']), train_positive=sum(y[i] == 1 for i in labeled['train']),
                sample_seed=6 + task, protocol_sha256=manifest['protocol_sha256']))
        if not included:
            raise ValueError('No valid tasks for ' + name)
        audit['datasets'][name] = dict(molecules=len(smiles), total_tasks=len(columns), prompt_tasks=len(prompts),
            raw_provenance=provenance, processed_smiles_sha256=digest(args.data_root / name / 'processed/smiles.csv'),
            included=included, excluded=excluded)
        print('{}: included={}/{}; exclusions={}'.format(name, len(included), len(columns), len(excluded)), flush=True)
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    (args.output_dir / 'audit.json').write_text(json.dumps(audit, indent=2, default=convert) + '\n')
    (args.output_dir / 'selected_prompt_downstream_task.json').write_text(json.dumps(selected, indent=2) + '\n')
    print('READY', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
