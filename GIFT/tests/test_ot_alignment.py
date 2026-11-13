import importlib

import pytest
import torch


@pytest.fixture
def ot(gift):
    return importlib.import_module("model.GIFT.GIMLETEncoderStack").compute_optimal_transport_plan


def test_marginals_and_zero_padding_mass(ot):
    torch.manual_seed(3)
    text = torch.randn(2, 5, 8, dtype=torch.float64)
    graph = torch.randn(2, 4, 8, dtype=torch.float64)
    text_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    graph_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    plan, _ = ot(text, graph, text_mask=text_mask, graph_mask=graph_mask, n_iters=300)
    valid_pairs = text_mask.unsqueeze(2) & graph_mask.unsqueeze(1)
    assert torch.count_nonzero(plan[~valid_pairs]) == 0
    torch.testing.assert_close(plan.sum(2), text_mask.double() / text_mask.sum(1, keepdim=True), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(plan.sum(1), graph_mask.double() / graph_mask.sum(1, keepdim=True), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("text_pad,graph_pad", [(0, 4), (5, 0), (5, 4)])
def test_transport_and_enhancement_do_not_depend_on_padding_or_batch(ot, text_pad, graph_pad):
    torch.manual_seed(9)
    text = torch.randn(1, 3, 8, dtype=torch.float64)
    graph = torch.randn(1, 2, 8, dtype=torch.float64)
    expected, _ = ot(text, graph)
    # Arbitrary, nonzero padded features must not consume any mass. The other
    # sample is longer, as in a batch whose longest molecule sets padding size.
    padded_text = torch.randn(2, 3 + text_pad, 8, dtype=torch.float64)
    padded_graph = torch.randn(2, 2 + graph_pad, 8, dtype=torch.float64)
    padded_text[0, :3], padded_graph[0, :2] = text[0], graph[0]
    text_mask = torch.ones(padded_text.shape[:2], dtype=torch.bool)
    graph_mask = torch.ones(padded_graph.shape[:2], dtype=torch.bool)
    text_mask[0, 3:], graph_mask[0, 2:] = False, False
    actual, _ = ot(padded_text, padded_graph, text_mask=text_mask, graph_mask=graph_mask)
    torch.testing.assert_close(actual[:1, :3, :2], expected)
    torch.testing.assert_close(torch.bmm(actual.transpose(1, 2), padded_text)[:1, :2],
                               torch.bmm(expected.transpose(1, 2), text))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_finite_gradients_and_no_padding_gradients(ot, dtype):
    torch.manual_seed(13)
    text = torch.randn(1, 4, 8, dtype=dtype, requires_grad=True)
    graph = torch.randn(1, 3, 8, dtype=dtype, requires_grad=True)
    plan, _ = ot(text, graph, text_mask=torch.tensor([[1, 1, 0, 0]]),
                 graph_mask=torch.tensor([[1, 1, 0]]))
    assert plan.dtype == dtype
    assert torch.isfinite(plan).all()
    (plan.float().square().sum()).backward()
    for features, valid_count in [(text, 2), (graph, 2)]:
        assert torch.isfinite(features.grad).all()
        assert features.grad[:, :valid_count].abs().sum() > 0
        assert torch.count_nonzero(features.grad[:, valid_count:]) == 0


def test_empty_side_and_wrong_mask_shape_are_rejected(ot):
    features = torch.ones(2, 3, 8)
    for side in ("text_mask", "graph_mask"):
        with pytest.raises(ValueError, match="at least one valid"):
            ot(features, features, **{side: torch.tensor([[1, 0, 0], [0, 0, 0]])})
        with pytest.raises(ValueError, match="must have shape"):
            ot(features, features, **{side: torch.ones(2, 4)})


def test_real_gift_forward_padding_invariance_and_backward(tiny_gift, gift_batch, monkeypatch):
    module = importlib.import_module("model.GIFT.GIMLETEncoderStack")
    original_ot = module.compute_optimal_transport_plan
    seen = []

    def checked_ot(*args, **kwargs):
        seen.append((kwargs["text_mask"].clone(), kwargs["graph_mask"].clone()))
        return original_ot(*args, **kwargs)

    monkeypatch.setattr(module, "compute_optimal_transport_plan", checked_ot)
    tiny_gift.eval()
    with torch.no_grad():
        expected = tiny_gift(**gift_batch()).logits
        actual = tiny_gift(**gift_batch(extra_graph_pad=2, extra_text_pad=3)).logits
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert len(seen) == 2 * len(tiny_gift.encoder.block)
    for text_mask, graph_mask in seen[-2:]:
        assert text_mask.shape == (2, 7)
        assert torch.equal(text_mask.sum(1), torch.tensor([4, 3]))
        # The virtual graph token and three real atoms all remain valid.
        assert torch.equal(graph_mask, torch.tensor([[1, 1, 1, 1, 0, 0]] * 2).bool())

    tiny_gift.train()
    loss = tiny_gift(**gift_batch(extra_graph_pad=2, extra_text_pad=3)).loss
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in tiny_gift.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    assert tiny_gift.encoder.block[0].attention.q_graph.weight.grad.abs().sum() > 0
    assert tiny_gift.encoder.graph_projector.layers[0].fc2.weight.grad.abs().sum() > 0
