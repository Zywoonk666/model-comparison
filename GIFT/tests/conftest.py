"""Small real GIFT models; only unrelated, eagerly imported baselines are stubbed."""
import importlib
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def gift(monkeypatch):
    pytest.importorskip("transformers")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    original_modules = set(sys.modules)

    def unused_baseline(*args, **kwargs):
        raise AssertionError("Only the GIFT backbone should run in these tests")

    for name, attributes in {
        "model.GIFT.gnn_model": ["GNN", "GNN_graphpred"],
        "model.KVPLM.kvplm_model": ["prepare_kvplm_model"],
        "model.MoMu.MoMu_Model": ["get_MoMu_model"],
        "model.llm": ["LLM"],
    }.items():
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


@pytest.fixture
def tiny_gift(gift):
    import torch
    from transformers import T5Config

    torch.manual_seed(17)
    config = T5Config(
        vocab_size=32, d_model=16, d_kv=8, d_ff=32,
        num_layers=2, num_decoder_layers=2, num_heads=2,
        dropout_rate=0.0, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    )
    graph_args = gift.GIMLETConfig()
    graph_args.transformer_backbone = "gimlet"
    graph_args.use_projector = True
    graph_args.use_ot_alignment = True
    graph_args.use_progressive_mask = True
    graph_args.dropout = 0.0
    graph_args.projector_dropout = 0.0
    model = gift.GraphT5TransformerForConditionalGeneration(config, graph_args)
    model.encoder.convert_to_graph_experts()
    return model


@pytest.fixture
def gift_batch():
    import torch
    from transformers import BatchEncoding

    def make(extra_graph_pad=0, extra_text_pad=0):
        n = 3 + extra_graph_pad
        x = torch.zeros(2, n, 2, dtype=torch.long)
        x[:, :3] = torch.tensor([[[2, 514], [3, 514], [4, 514]]] * 2)
        degree = torch.zeros(2, n, dtype=torch.long)
        degree[:, :3] = torch.tensor([[2, 3, 2]] * 2)
        bias = torch.full((2, n + 1, n + 1), float("-inf"))
        bias[:, :, :4] = 0
        spatial = torch.zeros(2, n, n, dtype=torch.long)
        spatial[:, :3, :3] = torch.tensor([[[1, 2, 3], [2, 1, 2], [3, 2, 1]]] * 2)
        edges = torch.zeros(2, n, n, 2, dtype=torch.long)
        edges[:, :3, :3] = 1
        edge_input = torch.zeros(2, n, n, 2, 2, dtype=torch.long)
        edge_input[:, :3, :3] = 1
        graph = BatchEncoding(dict(x=x, in_degree=degree, out_degree=degree.clone(),
                                   attn_bias=bias, spatial_pos=spatial,
                                   attn_edge_type=edges, edge_input=edge_input))
        ids = torch.tensor([[2, 3, 4, 1], [5, 6, 1, 0]])
        mask = ids.ne(0).long()
        return dict(graph=graph,
                    input_ids=torch.nn.functional.pad(ids, (0, extra_text_pad)),
                    attention_mask=torch.nn.functional.pad(mask, (0, extra_text_pad)),
                    labels=torch.tensor([[7, 1, -100], [8, 9, 1]]))
    return make
