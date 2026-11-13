import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mask_schedule import get_linear_mask_ratio


def test_one_based_endpoints_and_monotonic_decay():
    ratios = [get_linear_mask_ratio(epoch, 60) for epoch in range(1, 61)]
    assert ratios[0] == 1.0
    assert ratios[-1] == 0.0
    assert ratios[29] == pytest.approx(30 / 59)
    assert all(left > right for left, right in zip(ratios, ratios[1:]))
    assert get_linear_mask_ratio(1, 1, .8, .2) == .8
    assert [get_linear_mask_ratio(e, 3, .8, .2) for e in (1, 2, 3)] == pytest.approx([.8, .5, .2])
    assert all(get_linear_mask_ratio(e, 60, 1., 1.) == 1. for e in range(1, 61))


@pytest.mark.parametrize("epoch,total,start,end", [
    (0, 60, 1., 0.), (61, 60, 1., 0.), (1, 0, 1., 0.),
    (1, 60, 1.1, 0.), (1, 60, 1., -.1), (1, 60, .2, .8),
    (1, 60, float("nan"), 0.),
])
def test_invalid_schedule_is_rejected(epoch, total, start, end):
    with pytest.raises(ValueError):
        get_linear_mask_ratio(epoch, total, start, end)


@pytest.fixture
def update_epoch():
    # Execute the real training loop's schedule update without importing its
    # dataset/tokenizer entry point. The forward/backward below use real GIFT.
    tree = ast.parse((ROOT / "downstream_test.py").read_text())
    branch = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute)
                  and node.test.attr == "use_progressive_mask")
    code = compile(ast.Module(body=[branch], type_ignores=[]), "training_epoch_update", "exec")

    def update(model, epoch, total=60, start=1., end=0.):
        args = SimpleNamespace(use_progressive_mask=True, epochs=total,
                               mask_start_ratio=start, mask_end_ratio=end)
        exec(code, dict(args=args, model=model, epoch=epoch, get_linear_mask_ratio=get_linear_mask_ratio))
    return update


@pytest.fixture
def attention_biases(tiny_gift, monkeypatch):
    seen = []
    for block in tiny_gift.encoder.block:
        original = block.forward

        def record(*args, _original=original, **kwargs):
            seen.append(kwargs["position_bias"].detach().clone())
            return _original(*args, **kwargs)

        monkeypatch.setattr(block, "forward", record)
    return seen


def graph_reads_valid_text(bias, batch):
    graph_len = batch["graph"]["x"].shape[1] + 1
    region = bias[:, :, :graph_len, graph_len:]
    valid_graph = torch.cat((torch.ones(2, 1, dtype=torch.bool),
                             batch["graph"]["x"][:, :, 0].ne(0)), dim=1)
    valid_pairs = valid_graph[:, None, :, None] & batch["attention_mask"][:, None, None, :].bool()
    return region[valid_pairs.expand_as(region)]


def test_epoch_schedule_changes_actual_attention_and_keeps_padding_blocked(
        tiny_gift, gift_batch, update_epoch, attention_biases):
    batch = gift_batch(extra_graph_pad=2, extra_text_pad=3)
    observed = []
    for epoch in (1, 31, 60):
        update_epoch(tiny_gift, epoch)
        tiny_gift.train()
        tiny_gift.zero_grad()
        attention_biases.clear()
        loss = tiny_gift(**batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in tiny_gift.parameters() if p.grad is not None)
        assert len(attention_biases) == 2
        early, late = attention_biases
        observed.append(torch.isneginf(graph_reads_valid_text(early, batch)).float().mean().item())
        assert torch.isfinite(graph_reads_valid_text(late, batch)).all()
        for bias in attention_biases:
            graph_len = batch["graph"]["x"].shape[1] + 1
            # Last three text keys and two padded graph keys remain inaccessible
            # even at epoch 60, when the cross-modal curriculum reaches zero.
            assert (bias[:, :, :, -3:] < -1e4).all()
            assert torch.isneginf(bias[:, :, :, graph_len - 2:graph_len]).all()
            # Text queries may still read valid graph keys (the other direction).
            assert torch.isfinite(bias[:, :, graph_len:, :4]).all()
    assert observed[0] == 1.0
    assert 0.0 < observed[1] < 1.0
    assert observed[2] == 0.0


def test_training_loop_applies_custom_endpoints(tiny_gift, update_epoch):
    for epoch, expected in [(1, .8), (2, .5), (3, .2)]:
        update_epoch(tiny_gift, epoch, total=3, start=.8, end=.2)
        assert tiny_gift.encoder.graph_encoder.args.dynamic_mask_ratio == pytest.approx(expected)


def test_evaluation_keeps_original_deterministic_layerwise_mask(
        tiny_gift, gift_batch, update_epoch, attention_biases):
    batch = gift_batch()
    outputs = []
    for epoch in (1, 31, 60):
        update_epoch(tiny_gift, epoch)
        tiny_gift.eval()
        attention_biases.clear()
        with torch.no_grad():
            outputs.append(tiny_gift(**batch).logits)
        assert torch.isneginf(graph_reads_valid_text(attention_biases[0], batch)).all()
        assert torch.isfinite(graph_reads_valid_text(attention_biases[1], batch)).all()
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])


def test_disabling_progressive_mask_restores_strict_mask_in_all_layers(
        tiny_gift, gift_batch, attention_biases):
    tiny_gift.encoder.use_progressive_mask = False
    tiny_gift.encoder.graph_encoder.args.dynamic_mask_ratio = 0.0
    tiny_gift.train()
    batch = gift_batch()
    tiny_gift(**batch)
    for bias in attention_biases:
        assert torch.isneginf(graph_reads_valid_text(bias, batch)).all()
