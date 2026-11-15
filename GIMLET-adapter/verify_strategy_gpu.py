"""Check four real-checkpoint strategies with CUDA forward/backward/Adam steps."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

from offline_runtime import configure_offline_runtime
configure_offline_runtime()

import sklearn
import torch
from transformers import AutoTokenizer

from adapter_budget import STRATEGIES, adapter_spec
from model import GIMLETConfig, get_model
from model.GIMLET.adapter import configure_adapter_training, BottleneckAdapter
from dataloaders import MoleculeDatasetSplitLabel, add_prompt_transform_dict, graph_text_collator_dict
from run_adapter_strategies import REPO, WORKSPACE, idle_gpu, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    idle_gpu('0', 40.)
    assert torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained('t5-small', local_files_only=True)
    data = MoleculeDatasetSplitLabel(root=str(WORKSPACE / 'cache/property_data'), name='bace',
                                    single_split=0, rich_features=False)
    manifest = json.loads((WORKSPACE / 'GIFT/cache/graph2token_protocol_4090/bace/task_0_seed_6_shot_128.json').read_text())
    text = tokenizer(manifest['prompts'][0])
    labels = {1: tokenizer.convert_tokens_to_ids(tokenizer.tokenize('Yes')),
              0: tokenizer.convert_tokens_to_ids(tokenizer.tokenize('No')), 'invalid': [-100]}
    examples = []
    for item in manifest['splits']['train'][:16]:
        molecule = data[item['id']]
        examples.append(add_prompt_transform_dict['gimlet'](
            data=molecule, data_label=molecule.y, input_ids=text['input_ids'], attention_mask=text['attention_mask'],
            label_dict=labels, transform_in_collator=True, rich_features=False, tokenizer=tokenizer))
    collator = graph_text_collator_dict['gimlet'](tokenizer=tokenizer, transform_in_collator=True, rich_features=False)
    batch = {k: v.to('cuda') for k, v in collator(examples).items()}
    torch.cuda.set_per_process_memory_fraction(9.5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    records = []
    for strategy in STRATEGIES:
        torch.manual_seed(6)
        options = SimpleNamespace(transformer_backbone='gimlet', model_name_or_path='haitengzhao/gimlet',
                                  tokenizer_name='t5-small', few_shot_fashion='adapter', adapter_dim=64,
                                  adapter_strategy=strategy, adapter_placement='pre')
        graph_args = GIMLETConfig()
        graph_args.transformer_backbone = 'gimlet'
        model = get_model(options, graph_args, tokenizer).cuda().train()
        parameters = configure_adapter_training(model)
        spec = adapter_spec(model.config.d_model, 64, strategy, model.config.num_layers)
        assert sum(p.numel() for p in model.encoder.adapters.parameters()) == spec['adapter_parameters']
        optimizer = torch.optim.Adam(parameters, lr=.01)
        bottlenecks = [m for m in model.encoder.adapters.modules() if isinstance(m, BottleneckAdapter)]
        torch.cuda.reset_peak_memory_stats()
        losses = []
        for _ in range(2):
            optimizer.zero_grad()
            loss = model(**batch).loss
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
            # In particular, the LAST graph branch must affect the prediction.
            assert all(m.up.weight.grad.abs().sum().item() > 0 for m in bottlenecks)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            eval_loss = model(**batch).loss
            assert torch.isfinite(eval_loss)
        torch.cuda.synchronize()
        record = dict(strategy=strategy, adapter=spec, trainable=sum(p.numel() for p in parameters),
                      losses=losses, eval_loss=eval_loss.item(), batch_size=16,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                      gpu=torch.cuda.get_device_name(0), passed=True)
        records.append(record)
        print(json.dumps(record), flush=True)
        del model, parameters, optimizer, bottlenecks, loss, eval_loss
        gc.collect()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, records)


if __name__ == '__main__':
    main()
