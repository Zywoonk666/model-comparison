"""Export the *actual* GIFT support/query sets without changing its sampling."""
import hashlib
import json
import math
from pathlib import Path


def legacy_sample(indices, labels, shots, rng):
    """Match downstream_test.py, including integer bins and missing-label bins."""
    groups = {}
    for index in indices:
        index = int(index)
        label = float(labels[index])
        key = int(label) if math.isfinite(label) else -100
        groups.setdefault(key, []).append(index)
    sampled = []
    for group in groups.values():
        sampled.extend(rng.choice(group, size=min(len(group), shots), replace=False).tolist())
    return sampled


def export_task(output_dir, dataset, task, smiles, labels, splits, prompts, settings,
                source="gift_runtime"):
    """JSON carries IDs, SMILES, raw labels, prompts and training/eval settings."""
    rows = {}
    seen = set()
    for split, indices in splits.items():
        rows[split] = []
        for index in indices:
            index = int(index)
            if index in seen:
                raise ValueError("Duplicate or overlapping molecule ID: {}".format(index))
            seen.add(index)
            label = float(labels[index])
            label = label if math.isfinite(label) and label != -100 else None
            rows[split].append(dict(id=index, smiles=smiles[index], label=label))
    payload = dict(schema_version=1, protocol="gift_legacy", source=source,
                   dataset=dataset, task=int(task),
                   task_type="reg" if dataset in ("esol", "freesolv", "lipo") else "cla",
                   prompts=list(prompts), settings=settings, splits=rows)
    # Source is provenance, not a change in the actual experimental conditions.
    hashed = {key: value for key, value in payload.items() if key != "source"}
    payload["protocol_sha256"] = hashlib.sha256(
        json.dumps(hashed, sort_keys=True, allow_nan=False).encode()).hexdigest()
    target = Path(output_dir) / dataset / ("task_{}_seed_{}_shot_{}.json".format(
        task, settings["runseed"], settings["few_shot"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        old = json.loads(target.read_text())
        if old.get("protocol_sha256") != payload["protocol_sha256"]:
            raise ValueError("Protocol differs from existing export; use a new directory: {}".format(target))
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    return target


def settings_from_args(args):
    return dict(few_shot=args.few_shot, runseed=args.runseed, seed=args.seed,
                split=args.split, fractions=[0.8, 0.1, 0.1],
                batch_size=args.batch_size_ori, grad_accum_step=args.grad_accum_step,
                epochs=args.epochs, test_interval=args.test_interval,
                eval_epochs=[epoch for epoch in range(1, args.epochs + 1)
                             if (epoch + 1) % args.test_interval == 0],
                lr=args.lr, weight_decay=args.decay, optimizer="Adam",
                regression_metric="residual_std_ddof1")


def load_task(path):
    """Validate a shared manifest without importing ML dependencies."""
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1 or data.get("protocol") != "gift_legacy":
        raise ValueError("Unsupported comparison manifest: {}".format(path))
    hashed = {k: v for k, v in data.items() if k not in ("source", "protocol_sha256")}
    digest = hashlib.sha256(json.dumps(hashed, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if digest != data.get("protocol_sha256"):
        raise ValueError("Manifest checksum mismatch: {}".format(path))
    seen = set()
    for split in ("train", "valid", "test"):
        rows = data["splits"][split]
        if not rows:
            raise ValueError("Empty {} split".format(split))
        for row in rows:
            if type(row["id"]) is not int or row["id"] < 0 or row["id"] in seen:
                raise ValueError("Invalid/duplicate molecule ID or overlapping splits")
            seen.add(row["id"])
            label = row["label"]
            if label is not None and (not math.isfinite(label) or
                                      (data["task_type"] == "cla" and label not in (0, 1))):
                raise ValueError("Invalid task label")
        if split != "train" and data["task_type"] == "cla":
            if {row["label"] for row in rows if row["label"] is not None} != {0, 1}:
                raise ValueError("{} must contain both labeled classes for AUC".format(split))
    s = data["settings"]
    expected = [e for e in range(1, s["epochs"] + 1) if (e + 1) % s["test_interval"] == 0]
    if not expected or expected != s["eval_epochs"]:
        raise ValueError("Invalid GIFT evaluation schedule")
    return data


def apply_task(args, task, smiles, labels, prompts):
    """Resolve the saved IDs only after checking the local molecule table."""
    path = Path(args.comparison_protocol_dir) / args.dataset / "task_{}_seed_{}_shot_{}.json".format(
        task, args.runseed, args.few_shot)
    manifest = load_task(path)
    if manifest["dataset"] != args.dataset or manifest["task"] != task:
        raise ValueError("Manifest task identity mismatch")
    if manifest["settings"] != settings_from_args(args):
        raise ValueError("Training settings differ from the shared GIFT manifest")
    if manifest["prompts"] != list(prompts):
        raise ValueError("Task prompts differ from the shared GIFT manifest")
    indices = {}
    for split, rows in manifest["splits"].items():
        indices[split] = []
        for row in rows:
            index = row["id"]
            if index >= len(smiles) or smiles[index] != row["smiles"]:
                raise ValueError("Molecule order differs at ID {}; use matching raw data/RDKit".format(index))
            label = float(labels[index])
            label = label if math.isfinite(label) and label != -100 else None
            if label != row["label"]:
                raise ValueError("Label differs at molecule ID {}".format(index))
            indices[split].append(index)
    return manifest, indices


def select_validation_result(scores, task_type):
    if not scores or not all(math.isfinite(score) for score in scores):
        raise ValueError("Validation metrics are empty/nonfinite; cannot select a result")
    if task_type == "cla" and any(score <= 0 for score in scores):
        # Existing GIMLET evaluation uses 0 as the undefined-AUC sentinel.
        raise ValueError("Zero/undefined validation AUC; refusing test-based selection")
    best = max(scores) if task_type == "cla" else min(scores)
    return scores.index(best)


def result_metadata(manifest, best_index, prompt):
    rows = manifest["splits"]["train"]
    return dict(protocol_sha256=manifest["protocol_sha256"],
                metric="roc_auc" if manifest["task_type"] == "cla" else "residual_std_ddof1",
                few_shot=manifest["settings"]["few_shot"], n_train=len(rows),
                n_train_labeled=sum(row["label"] is not None for row in rows),
                best_epoch=manifest["settings"]["eval_epochs"][best_index],
                prompt_id=manifest["prompts"].index(prompt))
