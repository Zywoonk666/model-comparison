"""Launch GIFT/GIMLET variants using the same exported experimental conditions.

This launcher and --dry_run need only Python's standard library. Actual training
uses the active interpreter and each repository's original downstream pipeline.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from comparison_protocol import load_task

DATASETS = ["bace", "hiv", "bbbp", "tox21", "esol", "freesolv"]
VARIANTS = {"gift": "innovation", "gimlet_original": "linear", "gimlet_adapter": "adapter"}


def prepare(args):
    repo = args.repo.resolve()
    protocol_dir = args.protocol_dir.resolve()
    prompts = json.loads((repo / "prompts/selected_prompt_downstream_task.json").read_text())
    output = (args.output_dir or repo / "cache/baselines" / args.variant /
              "shot{}_seed{}".format(args.few_shot, args.runseed)).resolve()
    plans = []
    for dataset in args.datasets:
        expected_tasks = sorted(int(key) for key in prompts[dataset])
        manifests = []
        for task in expected_tasks:
            path = protocol_dir / dataset / "task_{}_seed_{}_shot_{}.json".format(
                task, args.runseed, args.few_shot)
            if not path.is_file():
                raise ValueError("Missing {}; run GIFT/export_graph2token_data.py first".format(path))
            manifest = load_task(path)
            if (manifest["dataset"], manifest["task"]) != (dataset, task):
                raise ValueError("Manifest task identity mismatch")
            if manifest["prompts"] != prompts[dataset][str(task)]:
                raise ValueError("{} task {} prompts differ".format(dataset, task))
            manifests.append(manifest)
        s = manifests[0]["settings"]
        if any(m["settings"] != s for m in manifests):
            raise ValueError("Task settings differ within {}".format(dataset))
        if s["few_shot"] != args.few_shot or s["runseed"] != args.runseed:
            raise ValueError("Manifest shot/seed mismatch")
        if s["split"] != "scaffold" or s["optimizer"] != "Adam":
            raise ValueError("Only the GIFT scaffold/Adam protocol is supported")
        command = [sys.executable, "downstream_test.py", "--transformer_backbone", "gimlet",
                   "--model_name_or_path", args.model_name_or_path, "--tokenizer_name", args.tokenizer_name,
                   "--dataset", dataset, "--few_shot_fashion", VARIANTS[args.variant],
                   "--few_shot", str(s["few_shot"]), "--runseed", str(s["runseed"]),
                   "--seed", str(s["seed"]), "--split", s["split"],
                   "--batch_size", str(s["batch_size"]), "--grad_accum_step", str(s["grad_accum_step"]),
                   "--lr", str(s["lr"]), "--decay", str(s["weight_decay"]),
                   "--epochs", str(s["epochs"]), "--test_interval", str(s["test_interval"]),
                   "--task_policy", "traversal", "--prompt_policy", "traversal",
                   "--prompt_file", "selected_prompt_downstream_task.json",
                   "--few_shot_prompt_fashion", "traversal", "--transform_in_collator", "--no_eval_train",
                   "--comparison_protocol_dir", str(protocol_dir),
                   "--output_result_to_file", str(output / (dataset + ".csv"))]
        if args.variant == "gimlet_adapter":
            command += ["--adapter_dim", str(args.adapter_dim)]
        elif args.variant == "gift":
            command += ["--use_projector", "True", "--use_ot_alignment", "True",
                        "--use_progressive_mask", "True", "--use_graph_expert", "True",
                        "--projector_dropout", "0.5", "--projector_blocks", "2", "--projector_act", "relu"]
        if args.data_root:
            command += ["--data_root", str(args.data_root.resolve())]
        plans.append(dict(dataset=dataset, command=command, settings=s,
                          manifests=[dict(task=m["task"], protocol_sha256=m["protocol_sha256"])
                                     for m in manifests]))
    return output, plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--protocol_dir", type=Path, default=Path(os.environ.get(
        "PROTOCOL_DIR", str(Path(__file__).resolve().parent / "cache/graph2token_protocol"))))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=os.environ.get("DATASETS", " ".join(DATASETS)).split())
    parser.add_argument("--few_shot", type=int, default=int(os.environ.get("FEW_SHOT", "128")))
    parser.add_argument("--runseed", type=int, default=int(os.environ.get("RUNSEED", "6")))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "0"), help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--model_name_or_path", default=os.environ.get("MODEL_NAME_OR_PATH", "haitengzhao/gimlet"))
    parser.add_argument("--tokenizer_name", default=os.environ.get("TOKENIZER_NAME", "t5-small"))
    parser.add_argument("--data_root", type=Path, default=os.environ.get("DATA_ROOT"))
    parser.add_argument("--output_dir", type=Path, default=os.environ.get("OUTPUT_DIR"))
    parser.add_argument("--adapter_dim", type=int, default=int(os.environ.get("ADAPTER_DIM", "64")))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    try:
        if args.adapter_dim < 1 or not args.datasets or len(set(args.datasets)) != len(args.datasets):
            raise ValueError("Use a positive adapter dimension and distinct nonempty datasets")
        output, plans = prepare(args)
        for plan in plans:
            print("{}: {} tasks; {}".format(plan["dataset"], len(plan["manifests"]),
                                           " ".join(shlex.quote(x) for x in plan["command"])), flush=True)
        if args.dry_run:
            print("PASS: manifests and prompts validated; training was not started.")
            return
        # Reject mixed/partial old results rather than silently resampling skipped tasks.
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty; choose a new --output_dir: {}".format(output))
        output.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.device
        env["PYTHONUNBUFFERED"] = "1"
        if env.get("HF_OFFLINE", "1") != "0":
            env.update(HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_OFFLINE="1")
        else:
            env.update(HF_DATASETS_OFFLINE="0", TRANSFORMERS_OFFLINE="0", HF_HUB_OFFLINE="0")
        (output / "run_config.json").write_text(json.dumps(dict(
            variant=args.variant, repo=str(args.repo.resolve()), device=args.device, plans=plans), indent=2) + "\n")
        for plan in plans:
            # Preserve terminal progress as well as a per-dataset log.
            with (output / (plan["dataset"] + ".log")).open("w") as log:
                process = subprocess.Popen(plan["command"], cwd=str(args.repo), env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                if process.wait():
                    raise RuntimeError("{} failed; see {}".format(plan["dataset"], log.name))
        print("Results:", output)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, "Error: {}\n".format(error))


if __name__ == "__main__":
    main()
