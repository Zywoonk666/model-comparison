import ast
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import comparison_protocol as protocol
import run_ablations as ablations


def setup_plans(tmp_path, monkeypatch, dataset="bace"):
    monkeypatch.setattr(ablations, "DATASETS", [dataset])
    settings = dict(few_shot=128, runseed=6, seed=42, split="scaffold", fractions=[.8, .1, .1],
                    batch_size=128, grad_accum_step=8, epochs=60, test_interval=20,
                    eval_epochs=[19, 39, 59], lr=.01, weight_decay=0., optimizer="Adam",
                    regression_metric="residual_std_ddof1")
    prompts = json.loads((ablations.REPO / "prompts/selected_prompt_downstream_task.json").read_text())
    for task, task_prompts in prompts[dataset].items():
        protocol.export_task(tmp_path / "protocol", dataset, int(task),
                             ["CC", "CO", "CN", "CCC", "CCO", "CCN"], [0., 1.] * 3,
                             dict(train=[0, 1], valid=[2, 3], test=[4, 5]), task_prompts, settings)
    return ablations.make_plans(tmp_path / "output", tmp_path / "protocol", tmp_path / "data")


def row_for(tmp_path, plan, condition, task, value):
    manifest = protocol.load_task(tmp_path / "protocol" / plan["dataset"] / "task_{}_seed_6_shot_128.json".format(task))
    flags = ablations.CONDITIONS[condition]
    return dict(dataset=plan["dataset"], split=task, prompt_id=0, runseed=6, epoch=60, few_shot=128,
                metric="roc_auc", protocol_sha256=manifest["protocol_sha256"], prompt=manifest["prompts"][0],
                model_name_or_path="haitengzhao/gimlet", model_variant="innovation", lr=.01,
                mask_schedule="linear_epoch" if flags["use_progressive_mask"] else "fixed",
                mask_start_ratio=flags["mask_start_ratio"] if flags["use_progressive_mask"] else 1.,
                mask_end_ratio=flags["mask_end_ratio"] if flags["use_progressive_mask"] else 1.,
                best_val_idx=0, best_epoch=19, valid_best=.7, test_best=value)


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_only_the_intended_component_changes(tmp_path, monkeypatch):
    plans = setup_plans(tmp_path, monkeypatch)
    full = ablations.comparable_command(plans["full"][0]["command"])
    expected = {"no_ot": "--use_ot_alignment", "no_progressive_mask": "--use_progressive_mask",
                "no_graph_expert": "--use_graph_expert", "fixed_epoch_mask": "--mask_end_ratio"}
    for condition, flag in expected.items():
        plan = plans[condition][0]
        command = ablations.comparable_command(plan["command"])
        assert [i for i in range(len(full)) if full[i] != command[i]] == [full.index(flag) + 1]
        assert plan["settings"] == plans["full"][0]["settings"]
        assert plan["manifests"] == plans["full"][0]["manifests"]
        assert command[command.index("--use_projector") + 1] == "True"


def test_rejects_wrong_protocol_and_duplicate_results(tmp_path, monkeypatch):
    plan = setup_plans(tmp_path, monkeypatch)["no_ot"][0]
    path = tmp_path / "rows.csv"
    row = row_for(tmp_path, plan, "no_ot", 0, .8)
    save_rows(path, [row])
    assert ablations.checked_rows(path, plan, "no_ot", tmp_path / "protocol", True)[1]
    row["protocol_sha256"] = "wrong"
    save_rows(path, [row])
    with pytest.raises(ValueError, match="mismatch"):
        ablations.checked_rows(path, plan, "no_ot", tmp_path / "protocol")
    row = row_for(tmp_path, plan, "no_ot", 0, .8)
    save_rows(path, [row, row])
    with pytest.raises(ValueError, match="Duplicate"):
        ablations.checked_rows(path, plan, "no_ot", tmp_path / "protocol")


@pytest.mark.parametrize("condition", list(ablations.CONDITIONS))
def test_validates_actual_downstream_mask_metadata(tmp_path, monkeypatch, condition):
    plan = setup_plans(tmp_path, monkeypatch)[condition][0]
    row = row_for(tmp_path, plan, condition, 0, .8)
    # Execute the real CSV writer's mask assignments so the test covers its
    # effective-value contract, including False -> fixed 1/1 despite CLI 1/0.
    tree = ast.parse((ablations.REPO / "downstream_test.py").read_text())
    statements = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
            target = node.targets[0]
            key = target.slice.value if isinstance(target.slice, ast.Index) else target.slice
            if (isinstance(target.value, ast.Name) and target.value.id == "df" and
                    isinstance(key, ast.Constant) and str(key.value).startswith("mask_")):
                statements.append(node)
    assert len(statements) == 4
    env = dict(df=row, args=SimpleNamespace(**ablations.CONDITIONS[condition]))
    exec(compile(ast.Module(body=statements, type_ignores=[]), "actual_csv_mask_metadata", "exec"), env)
    path = tmp_path / "actual.csv"
    save_rows(path, [row])
    assert ablations.checked_rows(path, plan, condition, tmp_path / "protocol", True)[1]
    if condition == "no_progressive_mask":
        assert row["mask_end_ratio"] == 1.
        row["mask_end_ratio"] = 0.
        save_rows(path, [row])
        with pytest.raises(ValueError, match="Training/selection mismatch"):
            ablations.checked_rows(path, plan, condition, tmp_path / "protocol", True)


def test_resume_adopts_complete_results_without_overwriting(tmp_path, monkeypatch):
    plans = setup_plans(tmp_path, monkeypatch)
    root = tmp_path / "output"
    for condition in ("full", "no_ot", "no_progressive_mask"):
        row = row_for(tmp_path, plans[condition][0], condition, 0, .8)
        save_rows(root / condition / "bace.csv", [row])
    config = dict(commit="test", device="0", python=sys.executable,
                  protocol_dir=str(tmp_path / "protocol"), data_root=str(tmp_path / "data"),
                  conditions=ablations.CONDITIONS, plans=plans,
                  reference=dict(csv_sha256=dict(bace=ablations.sha256(root / "full/bace.csv"))))
    old_status = dict(state="failed", started_at="original-start",
                      completed=[dict(condition="no_ot", dataset="bace", finished_at="original-finish")])
    ablations.write_json(root / "run_config.json", config)
    ablations.write_json(root / "status.json", old_status)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    _, completed = ablations.resume_completed(root, config)
    assert {(r["condition"], r["dataset"]) for r in completed} == {
        ("no_ot", "bace"), ("no_progressive_mask", "bace")}
    assert {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    assert next(r for r in completed if r["condition"] == "no_ot")["finished_at"] == "original-finish"
    (root / "no_graph_expert").mkdir()
    (root / "no_graph_expert/bace.log").write_text("unfinished training")
    with pytest.raises(ValueError, match="Partial job"):
        ablations.resume_completed(root, config)
    changed = dict(config, device="1")
    with pytest.raises(ValueError, match="configuration differs"):
        ablations.resume_completed(root, changed)


def test_tox21_mean_requires_all_twelve_tasks(tmp_path, monkeypatch):
    plans = setup_plans(tmp_path, monkeypatch, "tox21")
    root = tmp_path / "output"
    root.mkdir()
    ablations.write_json(root / "run_config.json", dict(plans=plans, protocol_dir=str(tmp_path / "protocol")))
    for condition in ("full", "no_ot"):
        plan = plans[condition][0]
        rows = [row_for(tmp_path, plan, condition, i, .5 + .01 * i) for i in range(12)]
        save_rows(root / condition / "tox21.csv", rows if condition == "full" else rows[:-1])
    assert ablations.summarize(root) is False
    with (root / "summary.csv").open() as stream:
        summary = next(csv.DictReader(stream))
    assert float(summary["full"]) == pytest.approx(.555)
    assert summary["no_ot"] == ""
    rows = [row_for(tmp_path, plans["no_ot"][0], "no_ot", i, .4 + .01 * i) for i in range(12)]
    save_rows(root / "no_ot/tox21.csv", rows)
    ablations.summarize(root)
    with (root / "deltas.csv").open() as stream:
        delta = next(csv.DictReader(stream))
    assert float(delta["degradation_vs_full"]) == pytest.approx(.1)
