"""Exercise real GIMLET/T5 layers using a tiny local checkpoint.

Only unused GIN/KVPLM/MoMu/LLM imports are replaced, because model/__init__.py
eagerly imports those baselines. No GIMLET, GraphEncoder, T5, or Adapter
computation is mocked. No model download or molecule dataset is required.
"""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def gimlet(monkeypatch):
    pytest.importorskip("transformers")
    original_modules = set(sys.modules)

    def unused_baseline(*args, **kwargs):
        raise AssertionError("This test must only execute the GIMLET backbone")

    imports = {
        "model.GIMLET.gnn_model": ["GNN", "GNN_graphpred"],
        "model.KVPLM.kvplm_model": ["prepare_kvplm_model"],
        "model.MoMu.MoMu_Model": ["get_MoMu_model"],
        "model.llm": ["LLM"],
    }
    for name, attributes in imports.items():
        module = ModuleType(name)
        for attribute in attributes:
            setattr(module, attribute, unused_baseline)
        monkeypatch.setitem(sys.modules, name, module)

    try:
        yield importlib.import_module("model")
    finally:
        for name in set(sys.modules) - original_modules:
            if name == "model" or name.startswith("model."):
                sys.modules.pop(name, None)


def test_pretrained_identity_training_and_checkpoint_reload(gimlet, tmp_path, monkeypatch):
    from transformers import BatchEncoding, T5Config
    from model.GIMLET.adapter import configure_adapter_training

    torch.manual_seed(17)
    graph_args = gimlet.GIMLETConfig()
    graph_args.transformer_backbone = "gimlet"
    config = T5Config(
        vocab_size=32, d_model=16, d_kv=8, d_ff=32,
        num_layers=2, num_decoder_layers=2, num_heads=2,
        dropout_rate=0.0, decoder_start_token_id=0, pad_token_id=0,
        eos_token_id=1,
    )
    cls = gimlet.GraphT5TransformerForConditionalGeneration
    baseline = cls(config, graph_args).eval()
    base_path = tmp_path / "base"
    baseline.save_pretrained(base_path)

    args = SimpleNamespace(
        transformer_backbone="gimlet", model_name_or_path="haitengzhao/gimlet",
        tokenizer_name=str(base_path), few_shot_fashion="adapter", adapter_dim=4,
    )
    # Exercise the official-checkpoint branch without network access, using
    # real from_pretrained() loading of the tiny local checkpoint instead.
    load_config = gimlet.AutoConfig.from_pretrained
    load_model = cls.from_pretrained
    with monkeypatch.context() as patch:
        patch.setattr(gimlet.AutoConfig, "from_pretrained", lambda *a, **kw: load_config(base_path))
        patch.setattr(cls, "from_pretrained", lambda *a, **kw: load_model(base_path, **kw))
        adapted = gimlet.get_model(args, graph_args, list(range(32))).eval()

    assert len(adapted.encoder.adapters) == 2
    assert adapted.config.adapter_dim == 4
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, adapted.state_dict()[name]), name
    for adapter in adapted.encoder.adapters:
        assert torch.count_nonzero(adapter.up.weight).item() == 0
        assert torch.count_nonzero(adapter.up.bias).item() == 0

    graph = BatchEncoding({
        "x": torch.tensor([[[2, 514], [3, 514], [4, 514]]] * 2),
        "in_degree": torch.tensor([[2, 3, 2]] * 2),
        "out_degree": torch.tensor([[2, 3, 2]] * 2),
        "attn_bias": torch.zeros(2, 4, 4),
        "spatial_pos": torch.tensor([[[1, 2, 3], [2, 1, 2], [3, 2, 1]]] * 2),
        "attn_edge_type": torch.ones(2, 3, 3, 2, dtype=torch.long),
        "edge_input": torch.ones(2, 3, 3, 2, 2, dtype=torch.long),
    })
    batch = dict(
        graph=graph,
        input_ids=torch.tensor([[2, 3, 4, 1], [5, 6, 1, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        labels=torch.tensor([[7, 1, -100], [8, 9, 1]]),
    )

    seen_shapes = []
    hooks = [adapter.register_forward_pre_hook(
        lambda module, inputs: seen_shapes.append(tuple(inputs[0].shape))
    ) for adapter in adapted.encoder.adapters]
    with torch.no_grad():
        expected = baseline(**batch).logits
        actual = adapted(**batch).logits
    for hook in hooks:
        hook.remove()
    assert seen_shapes == [(2, 8, 16), (2, 8, 16)]
    assert torch.equal(expected, actual)

    generation_batch = {key: value for key, value in batch.items() if key != "labels"}
    with torch.no_grad():
        assert torch.equal(
            baseline.generate(**generation_batch, max_length=4),
            adapted.generate(**generation_batch, max_length=4),
        )

    params = configure_adapter_training(adapted)
    frozen = {name: p.detach().clone() for name, p in adapted.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.Adam(params, lr=1e-3)
    adapted.train()
    for step in range(2):
        optimizer.zero_grad()
        loss = adapted(**batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        for adapter in adapted.encoder.adapters:
            assert adapter.up.weight.grad.abs().sum().item() > 0
            if step == 1:
                assert adapter.down.weight.grad.abs().sum().item() > 0
        optimizer.step()
    for name, p in adapted.named_parameters():
        if name in frozen:
            assert p.grad is None
            assert torch.equal(p, frozen[name]), name

    adapted.eval()
    saved_path = tmp_path / "adapted"
    adapted.save_pretrained(saved_path)
    restored, info = cls.from_pretrained(saved_path, output_loading_info=True)
    assert info["missing_keys"] == []
    assert info["unexpected_keys"] == []
    assert restored.config.adapter_dim == 4
    with torch.no_grad():
        assert torch.equal(adapted(**batch).logits, restored(**batch).logits)

    # Also exercise the existing local-checkpoint entry point with adapter mode.
    args.model_name_or_path = str(saved_path)
    restored_local = gimlet.get_model(args, graph_args, list(range(32))).eval()
    with torch.no_grad():
        assert torch.equal(adapted(**batch).logits, restored_local(**batch).logits)
