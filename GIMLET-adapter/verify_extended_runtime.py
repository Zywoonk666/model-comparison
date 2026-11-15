"""Exercise extended prompts/data and all five configurations on the real GPU."""
import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

from offline_runtime import configure_offline_runtime
configure_offline_runtime()
import torch
from transformers import AutoTokenizer
from dataloaders import MoleculeDatasetSplitLabel, add_prompt_transform_dict, graph_text_collator_dict
from model import GIMLETConfig, get_model
from model.GIMLET.adapter import configure_adapter_training
from run_adapter_strategies import WORKSPACE, idle_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol_dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    idle_gpu('0', 40.)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(9.5 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    tokenizer = AutoTokenizer.from_pretrained('t5-small', local_files_only=True)
    prompts = json.loads((args.protocol_dir / 'selected_prompt_downstream_task.json').read_text())
    label_dict = {1: tokenizer.convert_tokens_to_ids(tokenizer.tokenize('Yes')),
                  0: tokenizer.convert_tokens_to_ids(tokenizer.tokenize('No')), 'invalid': [-100]}
    records = []
    for dataset in ('cyp450', 'muv', 'toxcast'):
        task = max(prompts[dataset], key=lambda t: len(tokenizer(prompts[dataset][t][0])['input_ids']))
        manifest = json.loads((args.protocol_dir / dataset / ('task_%s_seed_6_shot_128.json' % task)).read_text())
        data = MoleculeDatasetSplitLabel(root=str(WORKSPACE / 'cache/property_data'), name=dataset,
                                        single_split=int(task), rich_features=False)
        sizes = data.slices['x'][1:] - data.slices['x'][:-1]
        ids = sorted([r['id'] for r in manifest['splits']['train']], key=lambda i: int(sizes[i]), reverse=True)[:16]
        text = tokenizer(manifest['prompts'][0])
        examples = [add_prompt_transform_dict['gimlet'](data=data[i], data_label=data[i].y,
            input_ids=text['input_ids'], attention_mask=text['attention_mask'], label_dict=label_dict,
            transform_in_collator=True, rich_features=False, tokenizer=tokenizer) for i in ids]
        collator = graph_text_collator_dict['gimlet'](tokenizer=tokenizer, transform_in_collator=True, rich_features=False)
        batch = {k: v.to('cuda') for k, v in collator(examples).items()}
        groups = [('shared', 'post'), ('shared', 'pre'), ('graph', 'pre'), ('text', 'pre'), ('separate', 'pre')] if dataset == 'cyp450' else [('shared', 'pre')]
        for strategy, placement in groups:
            torch.manual_seed(6)
            options = SimpleNamespace(transformer_backbone='gimlet', model_name_or_path='haitengzhao/gimlet',
                tokenizer_name='t5-small', few_shot_fashion='adapter', adapter_dim=64,
                adapter_strategy=strategy, adapter_placement=placement)
            graph_args = GIMLETConfig(); graph_args.transformer_backbone = 'gimlet'
            model = get_model(options, graph_args, tokenizer).cuda().train()
            params = configure_adapter_training(model)
            optimizer = torch.optim.Adam(params, lr=.01)
            torch.cuda.reset_peak_memory_stats()
            losses = []
            for _ in range(2):
                optimizer.zero_grad()
                loss = model(**batch).loss
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                optimizer.step(); losses.append(loss.item())
            model.eval()
            with torch.no_grad():
                inputs = {k: v for k, v in batch.items() if k != 'labels'}
                output = model.generate(**inputs, max_length=3, output_scores=True, return_dict_in_generate=True)
                assert torch.isfinite(output.scores[0]).all()
            record = dict(dataset=dataset, task=int(task), strategy=strategy, placement=placement,
                batch_size=len(ids), prompt_tokens=len(text['input_ids']), max_nodes=max(int(sizes[i]) for i in ids),
                losses=losses, peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2, passed=True)
            records.append(record); print(json.dumps(record), flush=True)
            del model, optimizer, params, loss, output, inputs
            gc.collect(); torch.cuda.empty_cache()
        del batch, data, examples, collator
        gc.collect(); torch.cuda.empty_cache()
    args.output.write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
