"""One real cached-checkpoint CUDA optimizer step for each ablation condition."""
import argparse
import gc
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import sklearn  # Initialize sklearn before the legacy model's eager imports.
import torch
from transformers import AutoTokenizer

from dataloaders import MoleculeDatasetSplitLabel, graph_text_collator_dict, add_prompt_transform_dict
from model import GIMLETConfig, get_model
from run_ablations import CONDITIONS, REPO, WORKSPACE, ensure_idle, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ensure_idle("0")
    assert torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained("t5-small", local_files_only=True)
    dataset = MoleculeDatasetSplitLabel(root=str(WORKSPACE / "cache/property_data"), name="bace",
                                       single_split=0, rich_features=False)
    prompt = json.loads((REPO / "prompts/selected_prompt_downstream_task.json").read_text())["bace"]["0"][0]
    text = tokenizer(prompt)
    labels = {1: tokenizer("Yes")["input_ids"], 0: tokenizer("No")["input_ids"], "invalid": [-100]}
    examples = [add_prompt_transform_dict["gimlet"](
        data=dataset[i], data_label=dataset[i].y, input_ids=text["input_ids"],
        attention_mask=text["attention_mask"], label_dict=labels,
        transform_in_collator=True, rich_features=False, tokenizer=tokenizer) for i in range(2)]
    collator = graph_text_collator_dict["gimlet"](tokenizer=tokenizer, transform_in_collator=True, rich_features=False)
    batch = {key: value.to("cuda") for key, value in collator(examples).items()}
    module = importlib.import_module("model.GIFT.GIMLETEncoderStack")
    results = []
    for condition, flags in CONDITIONS.items():
        torch.manual_seed(6)
        options = SimpleNamespace(transformer_backbone="gimlet", model_name_or_path="haitengzhao/gimlet",
                                  tokenizer_name="t5-small", few_shot_fashion="innovation", use_projector=True,
                                  projector_act="relu", projector_blocks=2, projector_dropout=0.5, **flags)
        graph_args = GIMLETConfig()
        graph_args.transformer_backbone = "gimlet"
        model = get_model(options, graph_args, tokenizer).cuda().train()
        assert model.encoder.use_ot_alignment == flags["use_ot_alignment"]
        assert model.encoder.use_progressive_mask == flags["use_progressive_mask"]
        assert model.encoder.use_graph_expert == flags["use_graph_expert"]
        model.encoder.graph_encoder.args.dynamic_mask_ratio = .5 if condition != "fixed_epoch_mask" else 1.
        model.requires_grad_(False)
        groups = [dict(params=list(model.lm_head.parameters()), lr=.01),
                  dict(params=list(model.encoder.graph_projector.parameters()), lr=.01)]
        experts = [p for name, p in model.named_parameters() if any(
            key in name for key in ("attention.q_graph", "attention.k_graph", "attention.v_graph", "ffn.graph_ffn"))]
        assert bool(experts) == flags["use_graph_expert"]
        if experts:
            groups.append(dict(params=experts, lr=.0001))
        for group in groups:
            for parameter in group["params"]:
                parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(groups, weight_decay=0.)
        with patch.object(module, "compute_optimal_transport_plan", wraps=module.compute_optimal_transport_plan) as ot:
            loss = model(**batch)["loss"]
            assert torch.isfinite(loss)
            loss.backward()
            assert ot.call_count == (len(model.encoder.block) if flags["use_ot_alignment"] else 0)
        for group in groups:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in group["params"])
        optimizer.step()
        model.eval()
        with torch.no_grad():
            eval_loss = model(**batch)["loss"]
            assert torch.isfinite(eval_loss)
        torch.cuda.synchronize()
        result = dict(condition=condition, train_loss=loss.item(), eval_loss=eval_loss.item(),
                      trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      graph_expert_parameters=sum(p.numel() for p in experts),
                      encoder_block=type(model.encoder.block[0]).__name__,
                      gpu=torch.cuda.get_device_name(0), passed=True)
        print(json.dumps(result), flush=True)
        results.append(result)
        del model, optimizer, groups, experts, loss, eval_loss, ot, parameter, group
        gc.collect()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, results)


if __name__ == "__main__":
    main()
