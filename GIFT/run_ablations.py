"""Run paired GIFT component ablations using the existing property protocol.

Planning and reporting use only the standard library. Training uses the same
downstream_test.py entry point as the completed full GIFT experiment.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import shutil
from statistics import mean
import subprocess
import sys
from types import SimpleNamespace

from comparison_protocol import load_task
from run_gimlet_baseline import DATASETS, prepare

REPO = Path(__file__).resolve().parent
WORKSPACE = REPO.parent
CONDITIONS = {
    "full": dict(use_ot_alignment=True, use_progressive_mask=True, use_graph_expert=True,
                 mask_start_ratio=1.0, mask_end_ratio=0.0),
    "no_ot": dict(use_ot_alignment=False, use_progressive_mask=True, use_graph_expert=True,
                  mask_start_ratio=1.0, mask_end_ratio=0.0),
    "no_progressive_mask": dict(use_ot_alignment=True, use_progressive_mask=False, use_graph_expert=True,
                                mask_start_ratio=1.0, mask_end_ratio=0.0),
    "no_graph_expert": dict(use_ot_alignment=True, use_progressive_mask=True, use_graph_expert=False,
                            mask_start_ratio=1.0, mask_end_ratio=0.0),
    "fixed_epoch_mask": dict(use_ot_alignment=True, use_progressive_mask=True, use_graph_expert=True,
                             mask_start_ratio=1.0, mask_end_ratio=1.0),
}


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_text(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value)
    temp.replace(path)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path, rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    write_text(path, buffer.getvalue())


def replace_option(command, name, value):
    command[command.index("--" + name) + 1] = str(value)


def comparable_command(command):
    command = list(command)
    replace_option(command, "output_result_to_file", "<OUTPUT>")
    return command


def make_plans(output, protocol, data):
    plans = {}
    for condition, flags in CONDITIONS.items():
        args = SimpleNamespace(repo=REPO, variant="gift", protocol_dir=Path(protocol),
                               output_dir=Path(output) / condition, datasets=DATASETS,
                               few_shot=128, runseed=6, model_name_or_path="haitengzhao/gimlet",
                               tokenizer_name="t5-small", data_root=Path(data),
                               mask_start_ratio=1.0, mask_end_ratio=0.0)
        _, plans[condition] = prepare(args)
        for plan in plans[condition]:
            for name, value in flags.items():
                replace_option(plan["command"], name, value)
    return plans


def checked_rows(path, plan, condition, protocol, require_complete=False):
    """Validate task identity/protocol/selection; never average incomplete Tox21."""
    expected = {}
    settings = plan["settings"]
    for task in plan["manifests"]:
        manifest = load_task(Path(protocol) / plan["dataset"] / "task_{}_seed_{}_shot_{}.json".format(
            task["task"], settings["runseed"], settings["few_shot"]))
        if manifest["protocol_sha256"] != task["protocol_sha256"]:
            raise ValueError("Protocol changed after planning")
        for prompt_id, prompt in enumerate(manifest["prompts"]):
            expected[(task["task"], prompt_id)] = (manifest, prompt)
    rows = []
    if Path(path).exists():
        with Path(path).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    seen = set()
    flags = CONDITIONS[condition]
    for row in rows:
        key = (int(row["split"]), int(row["prompt_id"]))
        if key in seen or key not in expected:
            raise ValueError("Duplicate/unexpected task in {}: {}".format(path, key))
        seen.add(key)
        manifest, prompt = expected[key]
        metric = "roc_auc" if manifest["task_type"] == "cla" else settings["regression_metric"]
        expected_fields = dict(dataset=plan["dataset"], runseed=str(settings["runseed"]),
                               epoch=str(settings["epochs"]), few_shot=str(settings["few_shot"]),
                               metric=metric, protocol_sha256=manifest["protocol_sha256"], prompt=prompt,
                               model_name_or_path="haitengzhao/gimlet", model_variant="innovation",
                               mask_schedule="linear_epoch" if flags["use_progressive_mask"] else "fixed")
        if any(row.get(k) != v for k, v in expected_fields.items()):
            raise ValueError("Result/protocol/condition mismatch in {}: {}".format(path, key))
        # downstream_test.py records effective ratios, not unused CLI values.
        start_ratio = flags["mask_start_ratio"] if flags["use_progressive_mask"] else 1.0
        end_ratio = flags["mask_end_ratio"] if flags["use_progressive_mask"] else 1.0
        if (float(row["lr"]) != settings["lr"] or
                float(row["mask_start_ratio"]) != start_ratio or
                float(row["mask_end_ratio"]) != end_ratio or
                int(row["best_epoch"]) != settings["eval_epochs"][int(row["best_val_idx"])]):
            raise ValueError("Training/selection mismatch in {}".format(path))
        if not all(math.isfinite(float(row[k])) for k in ("valid_best", "test_best")):
            raise ValueError("Nonfinite result in {}".format(path))
    complete = seen == set(expected)
    if require_complete and not complete:
        raise ValueError("Incomplete results in {}: {}/{}".format(path, len(seen), len(expected)))
    return rows, complete, len(expected)


def summarize(root):
    root = Path(root)
    config = json.loads((root / "run_config.json").read_text())
    summaries, details, progress, deltas = [], [], [], []
    for dataset_index, dataset in enumerate(DATASETS):
        metric = "residual_std_ddof1" if dataset in ("esol", "freesolv") else "roc_auc"
        summary = dict(dataset=dataset, metric=metric, **{name: "" for name in CONDITIONS})
        for condition in CONDITIONS:
            plan = config["plans"][condition][dataset_index]
            rows, complete, expected = checked_rows(root / condition / (dataset + ".csv"),
                                                   plan, condition, config["protocol_dir"])
            progress.append(dict(condition=condition, dataset=dataset, rows=len(rows),
                                 expected_rows=expected, complete=complete))
            for row in rows:
                details.append(dict(condition=condition, dataset=dataset, task=int(row["split"]),
                                    prompt_id=int(row["prompt_id"]), runseed=int(row["runseed"]),
                                    metric=metric, valid_best=float(row["valid_best"]),
                                    test_best=float(row["test_best"]), best_epoch=int(row["best_epoch"]),
                                    protocol_sha256=row["protocol_sha256"]))
            if complete:
                tasks = sorted({row["split"] for row in rows})
                summary[condition] = mean(mean(float(r["test_best"]) for r in rows if r["split"] == task)
                                          for task in tasks)
        summaries.append(summary)
        for condition in list(CONDITIONS)[1:]:
            if summary["full"] != "" and summary[condition] != "":
                difference = summary[condition] - summary["full"]
                deltas.append(dict(dataset=dataset, metric=metric, condition=condition,
                                   ablation_minus_full=difference,
                                   degradation_vs_full=-difference if metric == "roc_auc" else difference))
    write_csv(root / "summary.csv", summaries, ["dataset", "metric"] + list(CONDITIONS))
    write_csv(root / "results_long.csv", details, ["condition", "dataset", "task", "prompt_id", "runseed",
                                                 "metric", "valid_best", "test_best", "best_epoch", "protocol_sha256"])
    write_csv(root / "deltas.csv", deltas, ["dataset", "metric", "condition", "ablation_minus_full", "degradation_vs_full"])
    write_json(root / "progress.json", dict(updated_at=now(), datasets=progress))
    lines = ["# GIFT 消融结果", "", "分类：ROC-AUC ↑；回归：残差标准差（ddof=1）↓。Tox21 为完整 12 任务算术平均。",
             "完整模型复用已核验结果；各组 runseed=6。待完成单元格不参与比较。", "",
             "| 数据集 | full | no_ot | no_progressive_mask | no_graph_expert | fixed_epoch_mask |",
             "|---|---:|---:|---:|---:|---:|"]
    status_path = root / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        status_text = "队列状态：{}。实时进度见 status.json。".format(status["state"])
        if status.get("error"):
            status_text += " 原因：" + status["error"]
        lines[2:2] = [status_text, ""]
    for row in summaries:
        lines.append("| {} | {} |".format(row["dataset"], " | ".join(
            "待完成" if row[c] == "" else "{:.6f}".format(row[c]) for c in CONDITIONS)))
    lines += ["", "`deltas.csv` 中 degradation_vs_full > 0 表示消融后变差，< 0 表示消融后更好。",
              "这是单种子结果；去掉图专家也会减少可训练参数，不能据此声称统计显著或严格隔离参数量效应。", ""]
    write_text(root / "RESULTS.md", "\n".join(lines))
    return all(item["complete"] for item in progress)


def validate_reference(source, plans, protocol):
    source = Path(source)
    launch = json.loads((source.parent / "launch.json").read_text())
    if (source.parent / "exit_code").read_text().strip() != "0":
        raise ValueError("Reference run did not finish successfully")
    config = json.loads((source / "run_config.json").read_text())
    previous = {plan["dataset"]: plan for plan in config["plans"]}
    for plan in plans["full"]:
        old = previous[plan["dataset"]]
        if (old["settings"] != plan["settings"] or old["manifests"] != plan["manifests"] or
                comparable_command(old["command"]) != comparable_command(plan["command"])):
            raise ValueError("Reference command/protocol differs: " + plan["dataset"])
        checked_rows(source / (plan["dataset"] + ".csv"), plan, "full", protocol, require_complete=True)
    # The old run saved its tracked diff. Refuse reuse if runtime source differed.
    tracked = subprocess.check_output(["git", "ls-files", "GIFT"], cwd=str(WORKSPACE), text=True).splitlines()
    runtime = [p for p in tracked if (p.endswith(".py") and "/tests/" not in p) or
               p == "GIFT/prompts/selected_prompt_downstream_task.json"]
    old_diff = (source.parent / "working_tree.diff").read_text()
    if any("diff --git a/{} ".format(p) in old_diff for p in runtime):
        raise ValueError("Reference included runtime source changes; cannot certify reuse")
    subprocess.run(["git", "diff", "--quiet", launch["commit"], "--"] + runtime,
                   cwd=str(WORKSPACE), check=True)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    if sorted(freeze.splitlines()) != sorted((source.parent / "environment.freeze.txt").read_text().splitlines()):
        raise ValueError("Environment differs from the reference run")
    return dict(source=str(source), launch=launch,
                csv_sha256={d: sha256(source / (d + ".csv")) for d in DATASETS},
                runtime_sha256={p: sha256(WORKSPACE / p) for p in runtime}), freeze


def ensure_idle(device):
    processes = subprocess.check_output(["nvidia-smi", "--id=" + device,
        "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if processes:
        raise RuntimeError("GPU {} already has compute processes: {}".format(device, processes))


def resume_completed(root, config):
    """Only adopt complete validated jobs; never overwrite partial training."""
    old_config = json.loads((root / "run_config.json").read_text())
    for field in ("commit", "device", "python", "protocol_dir", "data_root", "conditions", "plans", "reference"):
        if old_config[field] != config[field]:
            raise ValueError("Resume configuration differs: " + field)
    old_status = json.loads((root / "status.json").read_text())
    if old_status["state"] not in ("failed", "completed"):
        raise ValueError("Resume requires a stopped queue with failed/completed status")
    previous = {(item["condition"], item["dataset"]): item for item in old_status["completed"]}
    completed = []
    for condition, plans in config["plans"].items():
        for plan in plans:
            path = root / condition / (plan["dataset"] + ".csv")
            if path.exists():
                checked_rows(path, plan, condition, config["protocol_dir"], require_complete=True)
                if condition == "full":
                    if sha256(path) != config["reference"]["csv_sha256"][plan["dataset"]]:
                        raise ValueError("Copied reference result changed")
                else:
                    completed.append(previous.get((condition, plan["dataset"]), dict(
                        condition=condition, dataset=plan["dataset"],
                        finished_at=datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),
                        recovered_at=now())))
            elif condition == "full" or (condition, plan["dataset"]) in previous:
                raise ValueError("Previously completed result is missing: " + str(path))
            elif path.with_suffix(".log").exists():
                raise ValueError("Partial job needs separate recovery; refusing to overwrite: " + str(path))
    return old_status, completed


def run(args):
    root = args.output_root.resolve()
    plans = make_plans(root, args.protocol_dir, args.data_root)
    reference, freeze = validate_reference(args.reuse_full.resolve(), plans, args.protocol_dir)
    config = dict(created_at=now(), branch=subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=str(WORKSPACE), text=True).strip(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(WORKSPACE), text=True).strip(),
        device=args.device, python=sys.executable, protocol_dir=str(args.protocol_dir.resolve()),
        data_root=str(args.data_root.resolve()), conditions=CONDITIONS, plans=plans,
        reference=reference, order="dataset first, then no_ot/no_progressive_mask/no_graph_expert/fixed_epoch_mask")
    if args.resume:
        old_status, completed = resume_completed(root, config)
    else:
        completed = []
    if args.dry_run:
        print(json.dumps(config, indent=2))
        print("PASS: all manifests, reference results, commands, runtime source and environment match.")
        return
    if root.exists() and not args.resume:
        raise ValueError("Output path already exists; choose a fresh path: " + str(root))
    lock_dir = WORKSPACE / "cache/ablations"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / ("gpu_" + args.device + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_idle(args.device)
        if args.resume:
            old_status, completed = resume_completed(root, config)
            archive = root / "recovery" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            archive.mkdir(parents=True)
            for filename in ("status.json", "exit_code", "finished_at", "pid", "RESULTS.md"):
                source = root / filename
                if source.exists():
                    shutil.copy2(source, archive / filename)
            shutil.copy2(__file__, archive / "run_ablations.snapshot.py")
            write_json(archive / "resume_config.json", config)
            for filename in ("exit_code", "finished_at"):
                if (root / filename).exists():
                    (root / filename).unlink()
            started_at = old_status["started_at"]
        else:
            root.mkdir(parents=True)
            write_json(root / "run_config.json", config)
            write_text(root / "environment.freeze.txt", freeze)
            write_text(root / "started_at", now() + "\n")
            shutil.copy2(__file__, root / "run_ablations.snapshot.py")
            for condition in CONDITIONS:
                folder = root / condition
                folder.mkdir()
                write_json(folder / "run_config.json", dict(condition=condition, flags=CONDITIONS[condition],
                           plans=plans[condition], reused_from=reference["source"] if condition == "full" else None))
            for dataset in DATASETS:
                source = args.reuse_full / (dataset + ".csv")
                if sha256(source) != reference["csv_sha256"][dataset]:
                    raise ValueError("Reference CSV changed")
                shutil.copy2(source, root / "full" / source.name)
            started_at = now()
        write_text(root / "pid", str(os.getpid()) + "\n")
        status = dict(state="running", started_at=started_at, launcher_pid=os.getpid(), completed=completed)
        if args.resume:
            status.update(resumed_at=now(), recovery_record=str(archive))
        completed_keys = {(item["condition"], item["dataset"]) for item in completed}
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=args.device, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[key] = "4"
        write_json(root / "status.json", status)
        summarize(root)
        write_text(WORKSPACE / "cache/latest_gift_ablation.txt", str(root) + "\n")
        try:
            with (root / "training.log").open("a" if args.resume else "w", buffering=1) as master:
                for index, dataset in enumerate(DATASETS):
                    for condition in list(CONDITIONS)[1:]:
                        if (condition, dataset) in completed_keys:
                            print("{} REUSE completed {} {}".format(now(), condition, dataset), flush=True)
                            continue
                        for path, digest in reference["runtime_sha256"].items():
                            if sha256(WORKSPACE / path) != digest:
                                raise ValueError("Runtime source changed during run: " + path)
                        ensure_idle(args.device)
                        plan = plans[condition][index]
                        title = "{} START {} {}: {}\n".format(now(), condition, dataset, shlex.join(plan["command"]))
                        print(title, end="", flush=True)
                        master.write(title)
                        with (root / condition / (dataset + ".log")).open("w", buffering=1) as log:
                            process = subprocess.Popen(plan["command"], cwd=str(REPO), env=env,
                                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                       universal_newlines=True)
                            status.update(current=dict(condition=condition, dataset=dataset,
                                                       pid=process.pid, started_at=now()))
                            write_json(root / "status.json", status)
                            for line in process.stdout:
                                log.write(line)
                                master.write(line)
                            code = process.wait()
                        if code:
                            raise RuntimeError("{} {} exited {}".format(condition, dataset, code))
                        checked_rows(root / condition / (dataset + ".csv"), plan, condition,
                                     args.protocol_dir, require_complete=True)
                        status["completed"].append(dict(condition=condition, dataset=dataset, finished_at=now()))
                        status.pop("current", None)
                        write_json(root / "status.json", status)
                        summarize(root)
                        print("{} DONE {} {}".format(now(), condition, dataset), flush=True)
            if not summarize(root):
                raise RuntimeError("Queue finished but results are incomplete")
            status.update(state="completed", finished_at=now(), exit_code=0)
        except Exception as error:
            status.update(state="failed", finished_at=now(), exit_code=1, error=str(error))
            raise
        finally:
            write_json(root / "status.json", status)
            write_text(root / "exit_code", str(status.get("exit_code", 1)) + "\n")
            write_text(root / "finished_at", now() + "\n")
            try:
                summarize(root)
            except Exception as report_error:
                # Keep the failure visible even if a malformed CSV blocks aggregation.
                path = root / "RESULTS.md"
                previous_report = path.read_text() if path.exists() else ""
                write_text(path, "队列状态：{}；错误：{}；汇总错误：{}\n\n{}".format(
                    status["state"], status.get("error", ""), report_error, previous_report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--reuse_full", type=Path)
    parser.add_argument("--protocol_dir", type=Path, default=REPO / "cache/graph2token_protocol_4090")
    parser.add_argument("--data_root", type=Path, default=WORKSPACE / "cache/property_data")
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue a stopped queue, preserving complete validated jobs")
    parser.add_argument("--summarize_only", action="store_true")
    args = parser.parse_args()
    try:
        if args.summarize_only:
            print("All complete:", summarize(args.output_root))
        elif args.reuse_full is None:
            parser.error("--reuse_full is required for a new ablation run")
        else:
            run(args)
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, "Error: {}\n".format(error))


if __name__ == "__main__":
    main()
