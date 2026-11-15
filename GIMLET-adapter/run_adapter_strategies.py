"""Compare four token Adapter strategies on identical property manifests.

One independent training process per job. Parallelism changes resource use,
not batch size, sampling, optimization, or checkpoint selection.
"""
import argparse
import csv
from datetime import datetime
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
import time
from types import SimpleNamespace

from adapter_budget import STRATEGIES, adapter_spec

REPO = Path(__file__).resolve().parent
WORKSPACE = REPO.parent
# Import protocol helpers without shadowing this repository's `model` package.
sys.path.append(str(WORKSPACE / "GIFT"))
from comparison_protocol import load_task
from run_gimlet_baseline import DATASETS, prepare


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_text(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    write_text(path, stream.getvalue())


def cached_config():
    hub = Path(os.environ.get("HF_HUB_CACHE", str(Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub")))
    model = hub / "models--haitengzhao--gimlet"
    revision = (model / "refs/main").read_text().strip()
    path = model / "snapshots" / revision / "config.json"
    return json.loads(path.read_text()), revision


def make_config(args, model_config=None, revision=None):
    if not 1 <= args.parallel <= 4 or args.memory_budget_gib <= 2:
        raise ValueError("Use parallel=1..4 and a GPU budget above 2 GiB")
    if len(set(args.datasets)) != len(args.datasets) or not args.datasets:
        raise ValueError("Dataset selection must be distinct and nonempty")
    if model_config is None:
        model_config, revision = cached_config()
    per_process = (args.memory_budget_gib - 2.) / args.parallel
    plans = {}
    for strategy in STRATEGIES:
        launch_args = SimpleNamespace(repo=REPO, variant="gimlet_adapter", protocol_dir=args.protocol_dir,
            output_dir=args.output_root / strategy, datasets=args.datasets, few_shot=128, runseed=6,
            model_name_or_path="haitengzhao/gimlet", tokenizer_name="t5-small", data_root=args.data_root,
            adapter_dim=args.adapter_dim)
        _, selected = prepare(launch_args)
        spec = adapter_spec(model_config["d_model"], args.adapter_dim, strategy, model_config["num_layers"])
        if spec["adapter_budget_gap_fraction"] > .01:
            raise ValueError("This reference width cannot match separate adapters within 1%; increase adapter_dim")
        head_count = model_config["vocab_size"] * model_config["d_model"]
        spec.update(adapter_placement="pre", head_parameters=head_count,
                    trainable_parameters=spec["adapter_parameters"] + head_count,
                    head_policy="tied_head_and_input_embeddings")
        for plan in selected:
            plan.update(strategy=strategy, adapter=spec, job_id=strategy + "/" + plan["dataset"])
            plan["command"] += ["--adapter_strategy", strategy, "--adapter_placement", "pre",
                                "--cuda_memory_limit_gib", str(per_process), "--disable_tqdm"]
            plans[(strategy, plan["dataset"])] = plan
    jobs = [plans[(strategy, dataset)] for dataset in args.datasets for strategy in STRATEGIES]
    return dict(created_at=now(), repo=str(REPO), output_root=str(args.output_root.resolve()),
                protocol_dir=str(args.protocol_dir.resolve()), data_root=str(args.data_root.resolve()),
                python=sys.executable, device=args.device, datasets=args.datasets, strategies=list(STRATEGIES),
                model_revision=revision, jobs=jobs, parallel=args.parallel,
                memory_budget_gib=args.memory_budget_gib, per_process_allocator_gib=per_process,
                context_margin_gib=2., threads_per_process=max(1, len(os.sched_getaffinity(0)) // args.parallel))


def checked_rows(root, job, protocol_dir):
    path = Path(root) / job["strategy"] / (job["dataset"] + ".csv")
    expected = {}
    settings = job["settings"]
    for task in job["manifests"]:
        manifest = load_task(Path(protocol_dir) / job["dataset"] / "task_{}_seed_{}_shot_{}.json".format(
            task["task"], settings["runseed"], settings["few_shot"]))
        if manifest["protocol_sha256"] != task["protocol_sha256"]:
            raise ValueError("Manifest changed after planning")
        for prompt_id, prompt in enumerate(manifest["prompts"]):
            expected[(task["task"], prompt_id)] = (manifest, prompt)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        key = (int(row["split"]), int(row["prompt_id"]))
        if key in seen or key not in expected:
            raise ValueError("Unexpected/duplicate task in " + str(path))
        seen.add(key)
        manifest, prompt = expected[key]
        metric = "roc_auc" if manifest["task_type"] == "cla" else settings["regression_metric"]
        fields = dict(dataset=job["dataset"], metric=metric, protocol_sha256=manifest["protocol_sha256"],
                      model_variant="adapter", model_name_or_path="haitengzhao/gimlet", prompt=prompt,
                      runseed=str(settings["runseed"]), epoch=str(settings["epochs"]),
                      few_shot=str(settings["few_shot"]))
        fields.update({k: str(v) for k, v in job["adapter"].items() if k != "adapter_budget_gap_fraction"})
        if any(row.get(k) != v for k, v in fields.items()):
            raise ValueError("Result configuration mismatch: " + str(path))
        idx = int(row["best_val_idx"])
        if not 0 <= idx < len(settings["eval_epochs"]) or int(row["best_epoch"]) != settings["eval_epochs"][idx]:
            raise ValueError("Invalid validation-selected epoch")
        if (float(row["lr"]) != settings["lr"] or
                not math.isclose(float(row["adapter_budget_gap_fraction"]), job["adapter"]["adapter_budget_gap_fraction"], abs_tol=1e-12)):
            raise ValueError("Learning rate or parameter budget differs")
        if int(row["n_train"]) != len(manifest["splits"]["train"]):
            raise ValueError("Training sample count differs")
        if not all(math.isfinite(float(row[k])) for k in ("test_best", "valid_best")):
            raise ValueError("Nonfinite score")
    if seen != set(expected):
        raise ValueError("Incomplete task results: {} ({}/{})".format(path, len(seen), len(expected)))
    return rows


def summarize(root, config, status):
    aggregates, details, comparisons = {}, [], []
    for job in config["jobs"]:
        if status["jobs"][job["job_id"]]["state"] != "completed":
            continue
        rows = checked_rows(root, job, config["protocol_dir"])
        tasks = sorted({int(row["split"]) for row in rows})
        aggregates[(job["strategy"], job["dataset"])] = mean(
            mean(float(r["test_best"]) for r in rows if int(r["split"]) == task) for task in tasks)
        for row in rows:
            details.append(dict(strategy=job["strategy"], dataset=job["dataset"], task=int(row["split"]),
                prompt_id=int(row["prompt_id"]), metric=row["metric"], test_best=float(row["test_best"]),
                valid_best=float(row["valid_best"]), best_epoch=int(row["best_epoch"]),
                adapter_parameters=int(row["adapter_parameters"]), trainable_parameters=int(row["trainable_parameters"]),
                protocol_sha256=row["protocol_sha256"]))
    for dataset in config["datasets"]:
        metric = "residual_std_ddof1" if dataset in ("esol", "freesolv") else "roc_auc"
        row = dict(dataset=dataset, metric=metric)
        row.update({s: aggregates.get((s, dataset), "") for s in STRATEGIES})
        comparisons.append(row)
    write_csv(root / "summary.csv", comparisons, ["dataset", "metric"] + list(STRATEGIES))
    write_csv(root / "results_long.csv", details, ["strategy", "dataset", "task", "prompt_id", "metric", "test_best",
              "valid_best", "best_epoch", "adapter_parameters", "trainable_parameters", "protocol_sha256"])
    lines = ["# Adapter 适配策略实验", "", "状态：{}；更新：{}。".format(status["state"], now()),
        "全部使用 pre-block 插入位置；同一参数预算上限；分离组误差低于 1%。",
        "四组均训练 Adapter 和共享 lm_head/input embedding；runseed=6，统一原始切分与训练设置。",
        "分类为 ROC-AUC ↑；回归沿用残差标准差 ddof=1 ↓，不是 RMSE。Tox21 跑齐 12 个任务后等权平均。", "",
        "| 数据集 | shared | graph | text | separate |", "|---|---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append("| {} | {} |".format(row["dataset"], " | ".join(
            "待完成" if row[s] == "" else "{:.6f}".format(row[s]) for s in STRATEGIES)))
    if status.get("error"):
        lines += ["", "队列停止原因：" + status["error"]]
    lines += ["", "实时进程与完成情况见 status.json；每组每数据集的日志和原始 CSV 位于对应子目录。",
              "这是单种子策略比较；旧的 post-block Adapter 结果不能直接当作本实验 shared 基线。", ""]
    write_text(root / "RESULTS.md", "\n".join(lines))


def idle_gpu(device, budget):
    pids = subprocess.check_output(["nvidia-smi", "--id=" + device,
        "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if pids:
        raise RuntimeError("GPU has existing compute processes; refusing to interfere: " + pids)
    total = float(subprocess.check_output(["nvidia-smi", "--id=" + device,
        "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True).strip()) / 1024.
    if budget > total - 1.:
        raise ValueError("GPU memory budget must leave at least 1 GiB outside the experiment")


def source_hashes():
    paths = [p for p in REPO.rglob("*.py") if not any(x in p.parts for x in ("cache", "tests", "__pycache__"))]
    paths += [WORKSPACE / "GIFT/comparison_protocol.py", WORKSPACE / "GIFT/run_gimlet_baseline.py",
              REPO / "prompts/selected_prompt_downstream_task.json"]
    return {str(p.relative_to(WORKSPACE)): digest(p) for p in sorted(paths)}


def save_state(root, config, status):
    status["updated_at"] = now()
    write_json(root / "status.json", status)
    summarize(root, config, status)


def run_queue(root, config, status):
    pending = [job for job in config["jobs"] if status["jobs"][job["job_id"]]["state"] != "completed"]
    running = {}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=config["device"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(config["threads_per_process"])
    try:
        save_state(root, config, status)
        while pending or running:
            while pending and len(running) < config["parallel"]:
                for path, expected in config["source_sha256"].items():
                    if digest(WORKSPACE / path) != expected:
                        raise ValueError("Runtime code changed during experiment: " + path)
                job = pending.pop(0)
                path = root / job["strategy"] / (job["dataset"] + ".log")
                log = path.open("x")
                try:
                    process = subprocess.Popen(job["command"], cwd=config["repo"], env=env,
                                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    raise
                running[job["job_id"]] = (job, process, log)
                status["jobs"][job["job_id"]].update(state="running", pid=process.pid, started_at=now())
                print("{} START {} PID={} {}".format(now(), job["job_id"], process.pid, shlex.join(job["command"])), flush=True)
                write_json(root / "status.json", status)
            for job_id, (job, process, log) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del running[job_id]
                entry = status["jobs"][job_id]
                entry.update(exit_code=code, finished_at=now())
                if code:
                    entry["state"] = "failed"
                    raise RuntimeError("{} exited {}; inspect its dataset log".format(job_id, code))
                try:
                    checked_rows(root, job, config["protocol_dir"])
                except Exception:
                    entry["state"] = "failed_validation"
                    raise
                entry["state"] = "completed"
                save_state(root, config, status)
                print("{} DONE {}".format(now(), job_id), flush=True)
            if running:
                time.sleep(2)
        status.update(state="completed", exit_code=0)
    except BaseException as error:
        status.update(state="stopped" if isinstance(error, KeyboardInterrupt) else "failed",
                      exit_code=130 if isinstance(error, KeyboardInterrupt) else 1,
                      error=str(error) or type(error).__name__)
        raise
    finally:
        for job, process, log in running.values():
            if process.poll() is None:
                process.terminate()
        for job, process, log in running.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
            status["jobs"][job["job_id"]].update(state="interrupted", exit_code=process.returncode, finished_at=now())
        status["finished_at"] = now()
        write_json(root / "status.json", status)
        write_text(root / "exit_code", str(status["exit_code"]) + "\n")
        write_text(root / "finished_at", status["finished_at"] + "\n")
        summarize(root, config, status)


def start(args):
    root = args.output_root.resolve()
    args.output_root = root
    config = make_config(args)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        print("PASS: manifests, prompts and parameter budgets validated; no GPU work started.")
        return
    lock_dir = WORKSPACE / "cache/adapter_strategies"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / ("gpu_" + args.device + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle_gpu(args.device, args.memory_budget_gib)
        hashes = source_hashes()
        if args.resume:
            old = json.loads((root / "run_config.json").read_text())
            status = json.loads((root / "status.json").read_text())
            if status["state"] not in ("failed", "stopped", "completed"):
                raise ValueError("Resume requires a stopped queue")
            for field in ("jobs", "model_revision", "parallel", "memory_budget_gib", "device", "protocol_dir", "data_root"):
                if config[field] != old[field]:
                    raise ValueError("Resume configuration differs: " + field)
            if hashes != old["source_sha256"]:
                raise ValueError("Runtime source differs; use a fresh experiment directory")
            current_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
            if sorted(current_freeze.splitlines()) != sorted((root / "environment.freeze.txt").read_text().splitlines()):
                raise ValueError("Python environment differs; use a fresh experiment directory")
            archive = root / "recovery" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            archive.mkdir(parents=True)
            for name in ("status.json", "exit_code", "finished_at", "pid"):
                if (root / name).exists():
                    shutil.copy2(root / name, archive / name)
            for job in config["jobs"]:
                entry = status["jobs"][job["job_id"]]
                if entry["state"] == "completed":
                    checked_rows(root, job, config["protocol_dir"])
                    continue
                # A dataset is the resume unit: preserve partial files, then
                # rerun its whole task traversal with the original seed.
                for suffix in (".csv", ".log"):
                    path = root / job["strategy"] / (job["dataset"] + suffix)
                    if path.exists():
                        dest = archive / job["strategy"] / path.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        path.replace(dest)
                status["jobs"][job["job_id"]] = dict(state="pending")
            config = old
            for key in ("exit_code", "finished_at", "error"):
                status.pop(key, None)
            for name in ("exit_code", "finished_at"):
                if (root / name).exists():
                    (root / name).unlink()
            status["resumed_at"] = now()
        else:
            if root.exists():
                raise ValueError("Choose a fresh output directory, or explicitly use --resume")
            root.mkdir(parents=True)
            for strategy in STRATEGIES:
                (root / strategy).mkdir()
            config.update(source_sha256=hashes, commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True).strip())
            write_json(root / "run_config.json", config)
            write_text(root / "environment.freeze.txt", subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True))
            shutil.copy2(__file__, root / "run_adapter_strategies.snapshot.py")
            budgets = [next(j["adapter"] for j in config["jobs"] if j["strategy"] == s) for s in STRATEGIES]
            write_csv(root / "parameter_budget.csv", budgets, list(budgets[0]))
            status = dict(started_at=now(), jobs={j["job_id"]: dict(state="pending") for j in config["jobs"]})
        status.update(state="running", launcher_pid=os.getpid())
        write_text(root / "pid", str(os.getpid()) + "\n")
        write_text(WORKSPACE / "cache/latest_adapter_strategy_run.txt", str(root) + "\n")
        run_queue(root, config, status)


def main():
    import signal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--protocol_dir", type=Path, default=WORKSPACE / "GIFT/cache/graph2token_protocol_4090")
    parser.add_argument("--data_root", type=Path, default=WORKSPACE / "cache/property_data")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--adapter_dim", type=int, default=64)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--memory_budget_gib", type=float, default=40.)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    args = parser.parse_args()

    def stop(signum, frame):
        raise KeyboardInterrupt("Stopped by signal {}".format(signum))

    signal.signal(signal.SIGTERM, stop)
    try:
        if args.summarize_only:
            summarize(args.output_root, json.loads((args.output_root / "run_config.json").read_text()),
                      json.loads((args.output_root / "status.json").read_text()))
        else:
            start(args)
    except (Exception, KeyboardInterrupt) as error:
        parser.exit(1, "Error: {}\n".format(error))


if __name__ == "__main__":
    main()
