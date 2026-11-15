import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from prepare_extended_benchmarks import corrected_prompts, labeled_support
from run_adapter_strategies import load_task
from comparison_protocol import export_task, settings_from_args, apply_task, select_validation_result


def test_independent_support_excludes_missing_and_preserves_pool():
    pool = list(range(400))
    before = list(pool)
    labels = np.array([0.] * 150 + [1.] * 130 + [np.nan] * 120)
    selected = labeled_support(pool, labels, 128, 6)
    assert pool == before
    assert len(selected) == len(set(selected)) == 256
    assert sum(labels[i] == 0 for i in selected) == 128
    assert sum(labels[i] == 1 for i in selected) == 128
    assert selected == labeled_support(pool, labels, 128, 6)
    rare = np.zeros(400); rare[399] = 1.
    assert 399 in labeled_support(pool, rare, 128, 7)


def test_cyp_prompt_reordering_matches_raw_columns_without_changing_source():
    source = json.loads((Path(__file__).resolve().parents[1] / 'prompts/selected_prompt_downstream_task.json').read_text())
    original = json.dumps(source)
    selected, mapping = corrected_prompts(source, 'cyp450',
        ['label_1a2', 'label_2c19', 'label_2c9', 'label_2d6', 'label_3a4'])
    assert 'particularly CYP2C19,' in selected['1'][0]
    assert 'particularly CYP2C9,' in selected['2'][0]
    assert mapping == {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}
    assert json.dumps(source) == original


def test_protocol_opt_in_and_invalid_labels(tmp_path):
    args = SimpleNamespace(few_shot=128, runseed=6, seed=42, split='scaffold', batch_size_ori=128,
        grad_accum_step=8, epochs=60, test_interval=20, lr=.01, decay=0.,
        comparison_sampling_policy='independent_labeled', comparison_protocol_dir=tmp_path, dataset='muv')
    settings = settings_from_args(args)
    smiles = ['CC', 'CO', 'CN', 'CCC', 'CCO', 'CCN']
    labels = [0., 1.] * 3
    splits = dict(train=[0, 1], valid=[2, 3], test=[4, 5])
    path = export_task(tmp_path, 'muv', 0, smiles, labels, splits, ['question'], settings,
                       protocol='independent_labeled_v1')
    assert load_task(path)['settings']['sampling_policy'] == 'independent_labeled'
    assert apply_task(args, 0, smiles, labels, ['question'])[1] == splits
    args.comparison_sampling_policy = 'legacy'
    assert 'sampling_policy' not in settings_from_args(args)
    with pytest.raises(ValueError, match='Training settings differ'):
        apply_task(args, 0, smiles, labels, ['question'])
    path = export_task(tmp_path / 'bad', 'muv', 0, smiles, [0., float('nan')] + labels[2:],
                       splits, ['question'], settings, protocol='independent_labeled_v1')
    with pytest.raises(ValueError, match='missing labels'):
        load_task(path)


def test_true_zero_auc_is_valid_only_with_audited_class_support():
    assert select_validation_result([0., .5, .3], 'cla', allow_zero_auc=True) == 1
    with pytest.raises(ValueError, match='Zero/undefined'):
        select_validation_result([0., .5, .3], 'cla')
    with pytest.raises(ValueError, match='AUC must'):
        select_validation_result([-.1, .5], 'cla', allow_zero_auc=True)
