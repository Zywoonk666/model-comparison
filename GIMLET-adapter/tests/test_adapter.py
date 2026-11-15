"""Check Adapter initialization and few-shot updates without dataset dependencies."""

import importlib.util
from pathlib import Path

import torch
from torch import nn
import pytest

from adapter_budget import adapter_spec, bottleneck_parameters


# Avoid model/__init__.py, which eagerly imports all optional baseline models.
spec = importlib.util.spec_from_file_location(
    "gimlet_adapter", Path(__file__).resolve().parents[1] / "model/GIMLET/adapter.py"
)
adapter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_module)
BottleneckAdapter = adapter_module.BottleneckAdapter
configure_adapter_training = adapter_module.configure_adapter_training
build_adapter = adapter_module.build_adapter


def test_initial_adapter_preserves_features_and_input_gradient():
    torch.manual_seed(7)
    adapter = BottleneckAdapter(d_model=16, bottleneck_dim=4)
    hidden = torch.randn(2, 7, 16, requires_grad=True)

    output = adapter(hidden)
    assert torch.equal(output, hidden)
    upstream = torch.randn_like(output)
    output.backward(upstream)
    assert torch.equal(hidden.grad, upstream)
    assert torch.count_nonzero(adapter.down.weight).item() > 0
    assert torch.count_nonzero(adapter.up.weight).item() == 0


class SmallTiedModel(nn.Module):
    """A small frozen backbone with T5-style tied input/output weights."""

    def __init__(self):
        super().__init__()
        self.shared = nn.Embedding(19, 16)
        self.encoder = nn.Module()
        self.encoder.backbone = nn.Linear(16, 16)
        self.encoder.adapters = nn.ModuleList([BottleneckAdapter(16, 4)])
        self.lm_head = nn.Linear(16, 19, bias=False)
        self.lm_head.weight = self.shared.weight

    def forward(self, ids):
        hidden = self.encoder.backbone(self.shared(ids))
        return self.lm_head(self.encoder.adapters[0](hidden))


def test_few_shot_updates_learn_from_zero_up_and_freeze_backbone():
    torch.manual_seed(11)
    model = SmallTiedModel()
    trainable = configure_adapter_training(model)
    expected = {id(p) for p in model.encoder.adapters.parameters()}
    expected.update(id(p) for p in model.lm_head.parameters())
    assert {id(p) for p in trainable} == expected
    assert len(trainable) == len(expected)
    assert model.shared.weight is model.lm_head.weight
    assert model.shared.weight.requires_grad

    frozen_before = {
        name: p.detach().clone() for name, p in model.named_parameters()
        if not p.requires_grad
    }
    adapter = model.encoder.adapters[0]
    down_before = adapter.down.weight.detach().clone()
    optimizer = torch.optim.SGD(trainable, lr=0.05)
    ids = torch.tensor([[2, 3, 4], [5, 6, 7]])
    labels = torch.tensor([8, 9, 10, 11, 12, 13])

    for step in range(2):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(ids).reshape(-1, 19), labels)
        loss.backward()
        assert torch.isfinite(loss)
        assert adapter.up.weight.grad.abs().sum().item() > 0
        if step == 0:
            assert torch.count_nonzero(adapter.down.weight.grad).item() == 0
        else:
            assert adapter.down.weight.grad.abs().sum().item() > 0
        optimizer.step()

    assert not torch.equal(down_before, adapter.down.weight)
    for name, p in model.named_parameters():
        if name in frozen_before:
            assert p.grad is None
            assert torch.equal(p, frozen_before[name])


@pytest.mark.parametrize('strategy', ['shared', 'graph', 'text', 'separate'])
def test_budget_and_direct_token_routing(strategy):
    module = build_adapter(512, 64, strategy)
    spec = adapter_spec(512, 64, strategy)
    assert sum(p.numel() for p in module.parameters()) == spec['adapter_parameters']
    assert spec['adapter_parameters'] <= bottleneck_parameters(512, 64)
    assert spec['adapter_budget_gap_fraction'] < .01
    hidden = torch.randn(2, 9, 512, requires_grad=True)
    assert torch.equal(module(hidden, graph_length=4), hidden)
    # Force a nonzero residual to test routing, not merely zero initialization.
    for child in module.modules():
        if isinstance(child, BottleneckAdapter):
            nn.init.constant_(child.up.bias, .25)
    out = module(hidden, graph_length=4)
    if strategy != 'text':
        assert torch.allclose(out[:, :4], hidden[:, :4] + .25)
    else:
        assert torch.equal(out[:, :4], hidden[:, :4])
    if strategy != 'graph':
        assert torch.allclose(out[:, 4:], hidden[:, 4:] + .25)
    else:
        assert torch.equal(out[:, 4:], hidden[:, 4:])
    out.sum().backward()
    assert torch.isfinite(hidden.grad).all()
    if strategy == 'separate':
        assert not ({id(p) for p in module.graph.parameters()} & {id(p) for p in module.text.parameters()})
        assert (spec['adapter_graph_dim'], spec['adapter_text_dim']) == (31, 31)


@pytest.mark.parametrize('boundary', [0, 9])
@pytest.mark.parametrize('strategy', ['graph', 'text', 'separate'])
def test_empty_token_slice_is_valid(strategy, boundary):
    module = build_adapter(16, 8, strategy)
    hidden = torch.randn(2, 9, 16)
    assert torch.equal(module(hidden, graph_length=boundary), hidden)
    with pytest.raises(ValueError, match='boundary'):
        module(hidden, graph_length=10)
