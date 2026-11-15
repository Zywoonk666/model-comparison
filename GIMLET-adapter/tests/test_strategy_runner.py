import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_adapter_strategies as runner
from comparison_protocol import export_task, load_task


def fixture_config(tmp_path, dataset='bace'):
    settings = dict(few_shot=128, runseed=6, seed=42, split='scaffold', fractions=[.8, .1, .1],
                    batch_size=128, grad_accum_step=8, epochs=60, test_interval=20, eval_epochs=[19, 39, 59],
                    lr=.01, weight_decay=0., optimizer='Adam', regression_metric='residual_std_ddof1')
    prompts = json.loads((runner.REPO / 'prompts/selected_prompt_downstream_task.json').read_text())
    for task, texts in prompts[dataset].items():
        export_task(tmp_path / 'protocol', dataset, int(task), ['CC', 'CO', 'CN', 'CCC', 'CCO', 'CCN'],
                    [0., 1.] * 3, dict(train=[0, 1], valid=[2, 3], test=[4, 5]), texts, settings)
    args = SimpleNamespace(output_root=tmp_path / 'out', protocol_dir=tmp_path / 'protocol',
                           data_root=tmp_path / 'data', parallel=4, memory_budget_gib=40.,
                           adapter_dim=64, datasets=[dataset], device='0')
    return runner.make_config(args, dict(d_model=512, num_layers=6, vocab_size=32100), 'test-revision')


def write_results(root, job, protocol_dir, tasks=None):
    rows = []
    for task in job['manifests'][:tasks]:
        manifest = load_task(Path(protocol_dir) / job['dataset'] / 'task_{}_seed_6_shot_128.json'.format(task['task']))
        rows.append(dict(job['adapter'], dataset=job['dataset'], split=task['task'], prompt_id=0,
                         prompt=manifest['prompts'][0], protocol_sha256=manifest['protocol_sha256'],
                         runseed=6, few_shot=128, epoch=60, lr=.01, metric='roc_auc', n_train=2,
                         model_variant='adapter', model_name_or_path='haitengzhao/gimlet',
                         best_epoch=39, best_val_idx=1, valid_best=.8, test_best=.5 + task['task']*.01))
    path = root / job['strategy'] / (job['dataset'] + '.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    runner.write_csv(path, rows, list(rows[0]))
    return path


def test_same_training_conditions_and_budget_with_four_parallel_jobs(tmp_path):
    config = fixture_config(tmp_path)
    assert config['per_process_allocator_gib'] == 9.5
    reference = config['jobs'][0]
    counts = {}
    for job in config['jobs']:
        assert job['settings'] == reference['settings']
        assert job['manifests'] == reference['manifests']
        command = job['command']
        for key, value in [('batch_size','128'),('grad_accum_step','8'),('adapter_placement','pre'),('runseed','6')]:
            assert command[command.index('--'+key)+1] == value
        counts[job['strategy']] = job['adapter']['adapter_parameters']
    assert counts == dict(shared=402816, graph=402816, text=402816, separate=399732)


def test_report_only_complete_tox21_and_reject_wrong_strategy(tmp_path):
    config = fixture_config(tmp_path, 'tox21')
    root = Path(config['output_root'])
    jobs = config['jobs']
    status = dict(state='running', jobs={j['job_id']: dict(state='running') for j in jobs})
    write_results(root, jobs[0], config['protocol_dir'])
    write_results(root, jobs[1], config['protocol_dir'], tasks=11)
    status['jobs'][jobs[0]['job_id']]['state'] = 'completed'
    runner.summarize(root, config, status)
    with (root / 'summary.csv').open() as f:
        row = next(csv.DictReader(f))
    assert float(row['shared']) == pytest.approx(.555)
    assert row['graph'] == ''
    with pytest.raises(ValueError, match='Incomplete'):
        runner.checked_rows(root, jobs[1], config['protocol_dir'])
    path = write_results(root, jobs[1], config['protocol_dir'])
    rows = list(csv.DictReader(path.open()))
    rows[0]['adapter_strategy'] = 'shared'
    runner.write_csv(path, rows, list(rows[0]))
    with pytest.raises(ValueError, match='configuration mismatch'):
        runner.checked_rows(root, jobs[1], config['protocol_dir'])


def test_supervisor_runs_and_validates_real_child_processes(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    root = Path(config['output_root'])
    worker = tmp_path / 'worker.py'
    worker.write_text('import pathlib,shutil,sys\nshutil.copyfile(sys.argv[1],sys.argv[2])\n')
    import sys
    for job in config['jobs']:
        source = write_results(tmp_path / 'fixtures', job, config['protocol_dir'])
        destination = root / job['strategy'] / 'bace.csv'
        destination.parent.mkdir(parents=True, exist_ok=True)
        job['command'] = [sys.executable, str(worker), str(source), str(destination)]
    config['source_sha256'] = {}
    config['repo'] = str(tmp_path)
    status = dict(state='running', jobs={j['job_id']: dict(state='pending') for j in config['jobs']})
    monkeypatch.setattr(runner.time, 'sleep', lambda n: None)
    runner.run_queue(root, config, status)
    assert status['state'] == 'completed'
    assert all(j['state'] == 'completed' for j in status['jobs'].values())
    assert (root / 'exit_code').read_text().strip() == '0'
    assert len({j['pid'] for j in status['jobs'].values()}) == 4


def test_offline_ogb_import_uses_real_version_without_update_threads(tmp_path):
    import os
    import subprocess
    import sys
    env = os.environ.copy()
    env['HF_HUB_OFFLINE'] = '1'
    probe = (
        'from offline_runtime import configure_offline_runtime\n'
        'assert configure_offline_runtime()\n'
        'import ogb,sys,threading\n'
        'from importlib.metadata import version\n'
        'assert ogb.__version__ == version("ogb")\n'
        'assert "outdated" not in sys.modules\n'
        'assert not any(t is not threading.main_thread() and not t.daemon for t in threading.enumerate())\n'
    )
    subprocess.run([sys.executable, '-c', probe], cwd=runner.REPO, env=env, check=True, timeout=15)
