import ast
import csv
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import comparison_protocol as protocol
from run_gimlet_baseline import prepare
from summarize_baselines import compare


def fixture_args(tmp_path):
    return SimpleNamespace(comparison_protocol_dir=str(tmp_path), dataset="bace", few_shot=128,
                           runseed=6, seed=42, split="scaffold", batch_size_ori=128,
                           batch_size=16, grad_accum_step=8, epochs=60, test_interval=20,
                           lr=.01, decay=0., num_workers=0)


def make_manifest(tmp_path):
    args = fixture_args(tmp_path)
    smiles = ["CC", "CO", "CN", "CCC", "CCO", "CCN", "CCl"]
    labels = [0., 1., 0., 1., 0., 1., float("nan")]
    path = protocol.export_task(tmp_path, "bace", 0, smiles, labels,
                                dict(train=[1, 6, 0], valid=[2, 3], test=[4, 5]),
                                ["Active?"], protocol.settings_from_args(args))
    return args, smiles, labels, path


def test_runtime_checks_samples_settings_prompts_and_hash(tmp_path):
    args, smiles, labels, path = make_manifest(tmp_path)
    manifest, indices = protocol.apply_task(args, 0, smiles, labels, ["Active?"])
    assert indices == dict(train=[1, 6, 0], valid=[2, 3], test=[4, 5])
    assert protocol.result_metadata(manifest, 1, "Active?")["best_epoch"] == 39
    assert protocol.result_metadata(manifest, 1, "Active?")["n_train_labeled"] == 2
    with pytest.raises(ValueError, match="Molecule order"):
        protocol.apply_task(args, 0, smiles[::-1], labels, ["Active?"])
    with pytest.raises(ValueError, match="Label differs"):
        protocol.apply_task(args, 0, smiles, [1.] + labels[1:], ["Active?"])
    with pytest.raises(ValueError, match="prompts differ"):
        protocol.apply_task(args, 0, smiles, labels, ["Changed?"])
    args.lr = .001
    with pytest.raises(ValueError, match="settings differ"):
        protocol.apply_task(args, 0, smiles, labels, ["Active?"])
    damaged = json.loads(path.read_text())
    damaged["settings"]["lr"] = .001
    path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="checksum"):
        protocol.load_task(path)


@pytest.mark.parametrize("repo", ["GIFT", "GIMLET-original", "GIMLET-adapter"])
def test_actual_downstream_manifest_branch_rebuilds_all_loaders(tmp_path, repo):
    args, smiles, labels, _ = make_manifest(tmp_path)
    tree = ast.parse((ROOT.parent / repo / "downstream_test.py").read_text())
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "args.comparison_protocol_dir"
                  and any(isinstance(n, ast.Name) and n.id == "comparison_indices" for n in ast.walk(node)))

    class Dataset:
        data = SimpleNamespace(y=np.asarray(labels).reshape(-1, 1))

        def __getitem__(self, ids):
            subset = Dataset()
            subset.ids = ids
            return subset

        def set_single_split(self, task):
            self.single_split = task

    def loader(dataset, **kwargs):
        return SimpleNamespace(dataset=dataset, kwargs=kwargs)

    env = dict(args=args, comparison_protocol=protocol, single_split_label=0,
               smiles_list=smiles, dataset=Dataset(), prompt={"0": ["Active?"]},
               DataLoader=loader, data_collator=object())
    exec(compile(ast.Module(body=[branch], type_ignores=[]), "runtime_branch", "exec"), env)
    assert env["train_index_total"] == [1, 6, 0]
    assert env["val_loader"].dataset.ids == [2, 3]
    assert env["test_loader"].dataset.ids == [4, 5]
    assert env["val_loader"].dataset.single_split == 0
    assert env["test_loader"].kwargs["shuffle"] is False


def test_validation_selection_never_falls_back_to_test():
    assert protocol.select_validation_result([.6, .9, .7], "cla") == 1
    assert protocol.select_validation_result([.5, 0., .3], "reg") == 1
    with pytest.raises(ValueError, match="undefined"):
        protocol.select_validation_result([0., 0.], "cla")
    with pytest.raises(ValueError, match="nonfinite"):
        protocol.select_validation_result([float("nan")], "reg")


@pytest.mark.parametrize("repo,variant,fashion", [
    ("GIFT", "gift", "innovation"), ("GIMLET-original", "gimlet_original", "linear"),
    ("GIMLET-adapter", "gimlet_adapter", "adapter")])
def test_launch_plan_uses_manifest_and_correct_model(tmp_path, repo, variant, fashion):
    source = json.loads((ROOT.parent / repo / "prompts/selected_prompt_downstream_task.json").read_text())
    args, smiles, labels, path = make_manifest(tmp_path)
    path.unlink()
    protocol.export_task(tmp_path, "bace", 0, smiles, labels,
                         dict(train=[1, 6, 0], valid=[2, 3], test=[4, 5]), source["bace"]["0"],
                         protocol.settings_from_args(args))
    args.repo, args.variant, args.protocol_dir = ROOT.parent / repo, variant, tmp_path
    args.output_dir, args.datasets, args.data_root = None, ["bace"], None
    args.model_name_or_path, args.tokenizer_name, args.adapter_dim = "haitengzhao/gimlet", "t5-small", 64
    output, plans = prepare(args)
    cmd = plans[0]["command"]
    assert cmd[cmd.index("--few_shot_fashion") + 1] == fashion
    assert cmd[cmd.index("--comparison_protocol_dir") + 1] == str(tmp_path)
    assert cmd[cmd.index("--runseed") + 1] == "6"
    assert cmd[cmd.index("--batch_size") + 1] == "128"
    assert cmd[cmd.index("--grad_accum_step") + 1] == "8"
    assert "--not_retest_tasks_in_result_file" not in cmd
    assert variant in str(output)
    assert plans[0]["manifests"][0]["protocol_sha256"]


def test_launcher_runs_from_repo_and_refuses_existing_results(tmp_path):
    args, smiles, labels, _ = make_manifest(tmp_path / "protocol")
    repo = tmp_path / "fake_repo"
    (repo / "prompts").mkdir(parents=True)
    (repo / "prompts/selected_prompt_downstream_task.json").write_text('{"bace":{"0":["Active?"]}}')
    # This subprocess is a launcher probe, not a model/performance test.
    (repo / "downstream_test.py").write_text(
        "import os,sys,json\nprint('LAUNCH_PROBE',os.getcwd(),os.environ['CUDA_VISIBLE_DEVICES'])\n")
    output = tmp_path / "results"
    cmd = [sys.executable, str(ROOT / "run_gimlet_baseline.py"), "--repo", str(repo),
           "--variant", "gimlet_original", "--protocol_dir", args.comparison_protocol_dir,
           "--datasets", "bace", "--device", "3", "--output_dir", str(output)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "LAUNCH_PROBE {} 3".format(repo) in (output / "bace.log").read_text()
    assert (output / "run_config.json").is_file()
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 1
    assert "not empty" in result.stderr


def test_summary_checks_protocol_and_averages_tasks(tmp_path):
    folders = {}
    for model in ("gift", "gimlet", "adapter", "graph2token"):
        folder = tmp_path / model
        folder.mkdir()
        folders[model] = folder
        rows = [dict(dataset="tox21", split=task, prompt_id=0, runseed=6,
                     protocol_sha256="task_{}".format(task), metric="roc_auc", test_best=score,
                     epoch=60, lr=.01) for task, score in [(0, .6), (1, .8)]]
        with (folder / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = compare(folders)
    assert len(result) == 1 and result[0]["n_tasks"] == 2
    assert result[0]["gift"] == pytest.approx(.7)
    target = folders["adapter"] / "results.csv"
    target.write_text(target.read_text().replace("task_0", "other_protocol"))
    with pytest.raises(ValueError, match="differs"):
        compare(folders)
