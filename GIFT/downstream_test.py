from os.path import join
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from sklearn.metrics import (r2_score,
                             roc_auc_score)
from dataloaders.splitters import random_scaffold_split, random_split, scaffold_split
from torch.utils.data import DataLoader

import plotly.graph_objects as go
import argparse

import commentjson
from basic_pipeline import load_graph_args,eval_result
from mask_schedule import get_linear_mask_ratio
from model import get_model
from dataloaders import add_prompt_transform_dict,\
    graph_text_collator_dict, \
    MoleculeDatasetSplitLabel,graph_text_tokenizer_dict
from transformers import AutoConfig
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
import transformers
transformers.utils.logging.set_verbosity_error()
from tqdm import tqdm
import os
import re
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()

# about seed and basic info
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runseed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--no_cuda',action='store_true')
parser.add_argument('--disable_tqdm',action='store_true')

# about dataset and dataloader
parser.add_argument('--use_huggingface_pipeline',action='store_true')
parser.add_argument('--dataset', type=str, default='bace')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--rich_features',action='store_true')
parser.add_argument('--transform_in_collator',action='store_true')
parser.add_argument('--overwrite_data_cache',action='store_true')

# about multitask strategies
parser.add_argument('--task_policy',type=str,default='traversal', choices=['single','traversal','multi_mixture','multi_label'])
parser.add_argument('--single_split',type=int,default=None)

#about model arguments
parser.add_argument('--tokenizer_name',type=str)
parser.add_argument('--model_name_or_path', type=str,)
parser.add_argument('--transformer_backbone',type=str,default='gimlet')
parser.add_argument('--return_model_size',action='store_true')

# about training strategies
parser.add_argument('--split', type=str, default='scaffold')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--grad_accum_step',type=int,default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_scale', type=float, default=1)
parser.add_argument('--decay', type=float, default=0)

# about testing strategies
parser.add_argument('--eval_train', dest='eval_train', action='store_true')
parser.add_argument('--no_eval_train', dest='eval_train', action='store_false')
parser.set_defaults(eval_train=True)
parser.add_argument('--only_test', action='store_true')
parser.add_argument('--test_interval', type=int, default=1)
parser.add_argument('--not_retest_tasks_in_result_file',action='store_true')
parser.add_argument('--plot_regression',action='store_true')

# about instruction strategies
parser.add_argument('--prompt_id',nargs='+', type=int,default=None)
parser.add_argument('--prompt_policy',type=str,default='traversal', choices=['single','sample','traversal'])
parser.add_argument('--prompt_augmentation',default='',choices=['','rewrite','expande','detail','shorten','name'])
parser.add_argument('--prompt_file',type=str,default='selected_prompt_downstream_task.json')

# about zero-shot strategies
parser.add_argument('--zero_shot',action='store_true')

#about few-shot strategies
parser.add_argument('--few_shot',type=int,default=None)
parser.add_argument('--few_shot_fashion',type=str,default='finetune',choices=['finetune', 'linear', 'innovation','prompttune'])
parser.add_argument('--few_shot_prompt_fashion',type=str,default='traversal',choices=['traversal', 'max','max_abs'])

#about saving strategies
parser.add_argument('--output_model_dir', type=str, default='')
parser.add_argument('--output_result_to_file',type=str,default=None)
parser.add_argument('--comparison_protocol_dir', type=str, default=None,
                    help='Use exact GIFT manifest sample IDs and validate experimental settings')
parser.add_argument('--data_root', type=str, default='property_data',
                    help='Local raw/processed molecule cache (use a compatible PyG version)')
parser.add_argument('--export_few_shot_dir', type=str, default=None,
                    help='Export actual sampled molecules and evaluation splits for Graph2Token')

# 在 parser.add_argument 列表里添加：
parser.add_argument("--use_projector", type=lambda x: (str(x).lower() == 'true'), default=True, help="Whether to use MLP projector")
parser.add_argument("--use_ot_alignment", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to use Optimal Transport (Sinkhorn) based cross-modal alignment")
parser.add_argument("--use_progressive_mask", type=lambda x: (str(x).lower() == 'true'), default=True, help="Enable layerwise relaxation and the training-epoch mask schedule")
parser.add_argument("--mask_start_ratio", type=float, default=1.0,
                    help="Training masking probability at epoch 1 (evaluation keeps fixed layerwise masking)")
parser.add_argument("--mask_end_ratio", type=float, default=0.0,
                    help="Training masking probability at the final epoch; equal endpoints give a constant mask")
parser.add_argument("--use_graph_expert", type=lambda x: (str(x).lower() == 'true'), default=True, help="Whether to use independent QKV and FFN for graph")
parser.add_argument("--graph_expert_lr_scale", type=float, default=0.01,
                    help="LR scale for graph-only QKV/FFN parameters in innovation tuning")
parser.add_argument("--projector_dropout", type=float, default=0.1, 
                    help="Dropout rate for the ResMLP projector blocks")

parser.add_argument('--projector_blocks', type=int, default=2, 
                    help='Number of residual blocks in the projector')
parser.add_argument('--projector_act', type=str, default='relu', choices=['gelu', 'relu'], 
                    help='Activation function for the projector (gelu or relu)')

args,left = parser.parse_known_args()
try:
    get_linear_mask_ratio(1, args.epochs, args.mask_start_ratio, args.mask_end_ratio)
except ValueError as error:
    parser.error(str(error))
if args.comparison_protocol_dir:
    if (args.use_huggingface_pipeline or args.split != 'scaffold' or args.few_shot is None
            or args.prompt_policy != 'traversal' or args.task_policy != 'traversal'
            or args.zero_shot or args.only_test or args.few_shot_prompt_fashion != 'traversal'
            or args.prompt_augmentation or args.not_retest_tasks_in_result_file):
        parser.error('Comparison requires fresh local scaffold few-shot task/prompt traversal')
    import importlib.util
    from pathlib import Path
    helper_path = Path(__file__).resolve().parent.parent / 'GIFT' / 'comparison_protocol.py'
    helper_spec = importlib.util.spec_from_file_location('comparison_protocol', str(helper_path))
    comparison_protocol = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(comparison_protocol)

if args.export_few_shot_dir and (args.use_huggingface_pipeline or args.split != 'scaffold'
                               or args.few_shot is None or args.prompt_policy != 'traversal'):
    parser.error('--export_few_shot_dir requires local scaffold few-shot with prompt traversal')
# print('arguments\t', args)

args,graph_args=load_graph_args(args,left)
# print("args:",args)
# print("graph_args:",graph_args)

if args.few_shot is not None:
    assert args.few_shot % args.batch_size ==0 or args.few_shot<args.batch_size
    if args.few_shot<args.batch_size:
        print('Warning: the few shot number is smaller than batch size; setting batch_size to few_shot')
        args.grad_accum_step=int(args.grad_accum_step/(args.batch_size/args.few_shot))
        args.batch_size=args.few_shot
        print('batch_size: ',args.batch_size,'grad_accum_step: ',args.grad_accum_step)


assert args.batch_size % args.grad_accum_step==0
args.batch_size_ori=args.batch_size
args.batch_size=args.batch_size_ori//args.grad_accum_step


if args.task_policy=='single':
    assert args.single_split is not None
else:
    if args.single_split is not None:
        print('single_split is specified as ',args.single_split,", but it will not be used in task_policy ",args.task_policy)

# Only when split and mixture all the labels, do split_label
args.split_label=args.task_policy=='multi'

def get_num_task(dataset):
    """ used in molecule_finetune.py """
    if dataset == 'tox21':
        return 12
    elif dataset in ['hiv', 'bace', 'bbbp', 'donor','esol','freesolv','lipo']:
        return 1
    elif dataset == 'pcba':
        return 128
    elif dataset == 'muv':
        return 17
    elif dataset == 'toxcast':
        return 617
    elif dataset == 'sider':
        return 27
    elif dataset == 'clintox':
        return 2
    elif dataset == 'cyp450':
        return 5
    raise ValueError(dataset + ': Invalid dataset name.')

def task_type(dataset):
    if dataset in ['esol','freesolv','lipo']:
        return 'reg'
    else:
        return 'cla'

def better_result(result,reference,dataset):
    if task_type(dataset)=='cla':
        return result>reference
    else:
        assert task_type(dataset)=='reg'
        return result<reference

def train(model, loader, optimizer):
    model.train()
    total_loss = 0

    for step, batch in tqdm(enumerate(loader),disable=args.disable_tqdm):
        for key in batch.keys():
            batch[key] = batch[key].to(model.device)

        loss= model(**batch)['loss']/args.grad_accum_step
        loss.backward()
        if (step+1) % args.grad_accum_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.detach().item()

    return total_loss / len(loader)*args.grad_accum_step

fig_number=0

def downstream_task_by_transform(model,train_loader,val_loader,test_loader,prompt=''):
    #reload the model parameter
    model = get_model(args, graph_args, tokenizer)
    model.to(device)
    if args.few_shot_fashion == 'finetune':
        model_param_group = [{'params': model.parameters()}]
    elif args.few_shot_fashion == 'linear':  #Only linear weight
        model.requires_grad_(False)
        if args.transformer_backbone=='kvplm':
            trainable_params = list(model.cls.predictions.decoder.parameters())
        elif args.transformer_backbone=='momu':
            trainable_params = list(model.text_proj_head[2].parameters())
        else:
            trainable_params = list(model.lm_head.parameters())
            for name, param in model.named_parameters():
                if 'graph_projector' in name or 'full_attn_res' in name:
                    trainable_params.append(param)
        for param in trainable_params:
            param.requires_grad_(True)
        model_param_group = [{'params': trainable_params}]
        #model_param_group = [{'params': model.lm_head.parameters()}]
    elif args.few_shot_fashion == 'innovation':
        model.requires_grad_(False)
        head_params = list(model.lm_head.parameters())
        projector_params = []
        graph_expert_params = []
        if args.use_projector:
            projector_params = list(model.encoder.graph_projector.parameters())
        if args.use_graph_expert:
            for name, param in model.named_parameters():
                if (
                    'attention.q_graph' in name
                    or 'attention.k_graph' in name
                    or 'attention.v_graph' in name
                    or 'ffn.graph_ffn' in name
                ):
                    graph_expert_params.append(param)
        trainable_params = head_params + projector_params + graph_expert_params
        for param in trainable_params:
            param.requires_grad_(True)
        model_param_group = [
            {'params': head_params, 'lr': args.lr},
            {'params': projector_params, 'lr': args.lr},
            {'params': graph_expert_params, 'lr': args.lr * args.graph_expert_lr_scale},
        ]
        model_param_group = [group for group in model_param_group if len(group['params']) > 0]
        print('Innovation tuning parameters: head={}, projector={}, graph_expert={}, graph_expert_lr={}'.format(
            sum(param.numel() for param in head_params),
            sum(param.numel() for param in projector_params),
            sum(param.numel() for param in graph_expert_params),
            args.lr * args.graph_expert_lr_scale
        ))
    elif args.few_shot_fashion == 'prompttune':
        model.requires_grad_(False)
        trainable_params = list(model.encoder.embed_tokens.parameters())
        for param in trainable_params:
            param.requires_grad_(True)
        model_param_group = [{'params': model.encoder.embed_tokens.parameters()}]
    else:
        raise ValueError("not supported few shot fashion")
    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print('Trainable parameters: {}'.format(trainable_count))
    optimizer = optim.Adam(model_param_group, lr=args.lr,
                        weight_decay=args.decay)

    train_roc_list, val_roc_list, test_roc_list = [], [], []
    train_acc_list, val_acc_list, test_acc_list = [], [], []

    train_full_list, val_full_list, test_full_list = [], [], []

    best_val_roc, best_val_idx = None, 0

    if args.zero_shot:
        if args.eval_train and (not args.only_test):
            train_roc, train_acc, train_target, train_pred = eval_result(model, train_loader,label_dict,tokenizer,task_type(args.dataset),args.transformer_backbone,args)
        else:
            train_roc={'score':0}
        if not args.only_test:
            val_roc, val_acc, val_target, val_pred = eval_result(model, val_loader,label_dict,tokenizer,task_type(args.dataset),args.transformer_backbone,args)
        else:
            val_roc={'score':0}
        test_roc, test_acc, test_target, test_pred = eval_result(model, test_loader,label_dict,tokenizer,task_type(args.dataset),args.transformer_backbone,args)

        print(
            'train: {}\tval: {}\ttest: {}'.format(train_roc, val_roc,
                                                                   test_roc))
        model_file=args.model_name_or_path if args.model_name_or_path is not None else graph_args.init_checkpoint
        if not(test_roc['score']==0) and args.output_result_to_file is not None:
            print('Outputing result to '+args.output_result_to_file)
            record = [(args.dataset,test_loader.dataset.single_split, model_file, train_roc['score'], val_roc['score'], test_roc['score'], prompt)]
            df = pd.DataFrame(record,
                              columns=['dataset', 'split','model_name_or_path', 'train_roc', 'val_roc', 'test_roc', 'prompt'])
            #Saved csv have an extra first column of index, which is always 0 in this case.
            file_name = args.output_result_to_file
            result_list = []
            if os.path.exists(file_name):
                result_list.append(pd.read_csv(file_name, index_col=0))
            result_list.append(df)
            result_per_dataset_table_permutated_all = pd.concat(result_list, ignore_index=True)
            result_per_dataset_table_permutated_all.to_csv(file_name, header=True)

            # df.to_csv(args.output_result_to_file, mode='a', header=False)

    else:
        for epoch in range(1, args.epochs + 1):
            if args.use_progressive_mask:
                current_mask_ratio = get_linear_mask_ratio(
                    epoch, args.epochs, args.mask_start_ratio, args.mask_end_ratio)
                model.encoder.graph_encoder.args.dynamic_mask_ratio = current_mask_ratio
                print(f"Epoch [{epoch:02d}/{args.epochs}] | 训练动态掩码率: {current_mask_ratio:.4f}")

            loss_acc = train(model, train_loader, optimizer)
            print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

            if (epoch+1) % args.test_interval==0:

                if args.eval_train:
                    train_roc, train_acc, train_target, train_pred = eval_result(model, train_loader,label_dict,tokenizer,task_type(args.dataset),args.transformer_backbone,args)
                else:
                    train_acc = 0
                    train_roc = {'score':0}
                val_roc, val_acc, val_target, val_pred = eval_result(model, val_loader,label_dict,tokenizer,task_type(args.dataset),args.transformer_backbone,args)
                test_roc, test_acc, test_target, test_pred = eval_result(model, test_loader,label_dict,tokenizer,task_type(args.dataset),args.transformer_backbone,args)

                train_roc_list.append(train_roc['score'])
                train_acc_list.append(train_acc)
                train_full_list.append(train_roc)
                val_roc_list.append(val_roc['score'])
                val_acc_list.append(val_acc)
                val_full_list.append(val_roc)
                test_roc_list.append(test_roc['score'])
                test_acc_list.append(test_acc)
                test_full_list.append(test_roc)
                print(
                    'train: {}\tval: {}\ttest: {}'.format(train_roc, val_roc,
                                                          test_roc))
                print()

                if best_val_roc is None:
                    best_val_roc = val_roc['score']
                    assert best_val_idx == 0

                if better_result(val_roc['score'], best_val_roc,args.dataset):
                    best_val_roc = val_roc['score']
                    best_val_idx = epoch - 1
                    if not args.output_model_dir == '':
                        output_model_path = join(args.output_model_dir, 'model_best.pth')
                        torch.save(model.state_dict(), output_model_path)

                        filename = join(args.output_model_dir, 'evaluation_best.pth')
                        np.savez(filename, val_target=val_target, val_pred=val_pred,
                                 test_target=test_target, test_pred=test_pred)
        if args.comparison_protocol_dir:
            best_val_idx = comparison_protocol.select_validation_result(val_roc_list, task_type(args.dataset))
        elif max(val_roc_list) > 0:
            best_val_idx=val_roc_list.index(max(val_roc_list)) if task_type(args.dataset)=='cla' else val_roc_list.index(min(val_roc_list))
        else:
            best_val_idx=test_roc_list.index(max(test_roc_list)) if task_type(args.dataset)=='cla' else val_roc_list.index(min(test_roc_list))

        print(
            'best train: {}\tval: {}\ttest: {}'.format(train_full_list[best_val_idx], val_full_list[best_val_idx],
                                                                   test_full_list[best_val_idx]))

        model_file=args.model_name_or_path if args.model_name_or_path is not None else graph_args.init_checkpoint
        if (args.comparison_protocol_dir or test_roc_list[best_val_idx] != 0) and args.output_result_to_file is not None:
            print('Outputing result to file '+args.output_result_to_file)
            record=[(args.dataset,test_loader.dataset.single_split,model_file,args.epochs,args.lr,args.runseed,best_val_idx,train_roc_list[best_val_idx],val_roc_list[best_val_idx],test_roc_list[best_val_idx],prompt)]
            df = pd.DataFrame(record,
                              columns=['dataset','split', 'model_name_or_path','epoch','lr','runseed','best_val_idx','train_best','valid_best','test_best','prompt'
                                       ])
            df['mask_schedule'] = 'linear_epoch' if args.use_progressive_mask else 'fixed'
            df['mask_start_ratio'] = args.mask_start_ratio if args.use_progressive_mask else 1.0
            df['mask_end_ratio'] = args.mask_end_ratio if args.use_progressive_mask else 1.0
            df['mask_eval_ratio'] = 1.0

            if args.comparison_protocol_dir:
                for key, value in comparison_protocol.result_metadata(comparison_manifest, best_val_idx, prompt).items():
                    df[key] = value
                df['model_variant'] = args.few_shot_fashion

            file_name = args.output_result_to_file
            result_list = []
            if os.path.exists(file_name):
                result_list.append(pd.read_csv(file_name, index_col=0))
            result_list.append(df)
            result_per_dataset_table_permutated_all = pd.concat(result_list, ignore_index=True)
            result_per_dataset_table_permutated_all.to_csv(file_name, header=True)

        if args.output_model_dir is not '':
            output_model_path = join(args.output_model_dir, 'model_final.pth')
            torch.save(model.state_dict(), output_model_path)



if __name__ == '__main__':
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device('cuda:' + str(args.device)) \
        if (torch.cuda.is_available() and not args.no_cuda) else torch.device('cpu')
    if torch.cuda.is_available() and not args.no_cuda :
        torch.cuda.manual_seed_all(args.runseed)

    tokenizer_kwargs = {
        "cache_dir": None,
        "use_fast": True,
        "revision": 'main',
        "use_auth_token": None,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, **tokenizer_kwargs)

    model=get_model(args,graph_args,tokenizer)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Total Trainable Model size: {}'.format(count_parameters(model)))
    

    if not args.use_huggingface_pipeline:
        #Load instruction files, and add them for molecule data.
        if args.few_shot and args.few_shot_prompt_fashion!='traversal':

            def modify_name(name):
                name = name.replace('.ckpt', '.pt')
                name=name.replace('ckpts/','')
                if name[-1]=='/':
                    name=name[:-1]
                return name

            file_name=os.path.join('cache','result_'+args.few_shot_prompt_fashion+'_prompt_table.csv')
            prompts_pd = pd.read_csv(file_name,index_col='unique_task_id')
            rename_keys={}
            for name in prompts_pd.columns:
                rename_keys[name]=modify_name(name)
            prompts_pd=prompts_pd.rename(columns=rename_keys)
            prompt={}
            model_name=modify_name(args.model_name_or_path)
            for ind in range(get_num_task(args.dataset)):
                if args.dataset + '@' + str(ind) in prompts_pd.index.values:
                    res=prompts_pd.loc[args.dataset+'@'+str(ind),model_name]
                    if pd.isna(res):
                        continue
                    prompt[str(ind)]=[res]

        else:
            if args.prompt_augmentation=='':
                with open(os.path.join("prompts",args.prompt_file), 'r') as load_f:
                    prompts = commentjson.load(load_f)
                prompt=prompts[args.dataset]
            else:
                with open(os.path.join("prompts",args.prompt_file), 'r') as load_f:
                    prompts = commentjson.load(load_f)
                prompt_all=prompts[args.dataset]
                prompt={}
                for key in prompt_all:
                    if args.prompt_augmentation in prompt_all[key]:
                        prompt[key]=prompt_all[key][args.prompt_augmentation]
                    else:
                        print('label split {} has no augmentation {}'.format(key, args.prompt_augmentation))

        if isinstance(prompt,list):
            prompt_token=tokenizer(prompt,return_special_tokens_mask=True)
            input_ids = [item for item in prompt_token.data['input_ids']]
            attention_mask = [item for item in prompt_token.data['attention_mask']]
            if args.prompt_id is None:
                args.prompt_id = list(range(len(prompt)))
        elif isinstance(prompt,dict):
            prompt_token={}
            input_ids={}
            attention_mask={}
            args.prompt_id={}
            for key in prompt.keys():
                if len(prompt[key])>0:
                    prompt_token[key]=tokenizer(prompt[key],return_special_tokens_mask=True)
                    input_ids[key] = [item for item in prompt_token[key].data['input_ids']]
                    attention_mask[key] = [item for item in prompt_token[key].data['attention_mask']]
                    args.prompt_id[key] = list(range(len(prompt[key])))
        else:
            raise ValueError('Prompt type not supported. Only list or dict of (list of) prompts are supported.')

    else:
        print('Using huggingface pipeline. Prompt file not loaded.')

    label_ignore = [-100]
    raw_label = {1: 'Yes', 0: 'No', 'invalid': label_ignore}
    label_y = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_label[1])) # Not include CLS or other tokens
    label_n = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_label[0]))
     # input a list so that they can be concatenated in collator
    label_dict = {1: label_y, 0: label_n, 'invalid': label_ignore}

    # Bunch of classification tasks
    num_tasks = get_num_task(args.dataset)

    if not args.use_huggingface_pipeline:
        # Loading Molecule Dataset
        dataset_folder = os.path.join(args.data_root, '')

        if args.transformer_backbone in ['kvplm', 'galactica','gpt3']:
            dataset = MoleculeDatasetSplitLabel(root=dataset_folder, name=args.dataset,return_smiles=True,split_label=args.split_label,single_split=args.single_split,rich_features=args.rich_features)
        else:
            dataset = MoleculeDatasetSplitLabel(root=dataset_folder, name=args.dataset,split_label=args.split_label,single_split=args.single_split,rich_features=args.rich_features)

        print(dataset)
        print(dataset[0])

        if args.split == 'scaffold':
            # if args.single_split is not None:
            smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                      header=None)[0].tolist()
            train_index, valid_index, test_index = scaffold_split(
                torch.arange(len(smiles_list)), smiles_list, null_value=0, frac_train=0.8,
                frac_valid=0.1, frac_test=0.1)

            train_index_total=[]
            valid_index_total=[]
            test_index_total=[]
            for times in range(dataset.label_number):
                train_index_times=train_index+times*dataset.len_oridata()
                valid_index_times = valid_index + times * dataset.len_oridata()
                test_index_times = test_index + times * dataset.len_oridata()

                train_index_total.append(train_index_times)
                valid_index_total.append(valid_index_times)
                test_index_total.append(test_index_times)
            train_index_total=torch.cat(train_index_total,0)
            valid_index_total=torch.cat(valid_index_total,0)
            test_index_total=torch.cat(test_index_total,0)

            train_dataset = dataset[train_index_total]
            valid_dataset = dataset[valid_index_total]
            test_dataset = dataset[test_index_total]

            print('split via scaffold')
        elif args.split == 'random':
            train_dataset, valid_dataset, test_dataset = random_split(
                dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
                frac_test=0.1, seed=args.seed)
            print('randomly split')
        elif args.split == 'random_scaffold':
            smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                      header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = random_scaffold_split(
                dataset, smiles_list, null_value=0, frac_train=0.8,
                frac_valid=0.1, frac_test=0.1, seed=args.seed)
            print('random scaffold')
        else:
            raise ValueError('Invalid split option.')
        print(train_dataset[0])

        data_collator = graph_text_collator_dict[args.transformer_backbone](
            tokenizer=tokenizer,
            transform_in_collator=args.transform_in_collator,
            rich_features=args.rich_features)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,collate_fn=data_collator)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,collate_fn=data_collator)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,collate_fn=data_collator)

    else:
        # Loading Huggingface Dataset
        dataset = load_dataset("haitengzhao/molecule_property_instruction",
                               # download_mode = "force_redownload"
                               )[args.dataset]


        print(dataset)
        print(dataset[0])

        if args.split == 'scaffold':
            train_dataset_total = dataset.filter(lambda example: (example["split"] == 'train'))
            valid_dataset_total = dataset.filter(lambda example: (example["split"] == 'valid'))
            test_dataset_total = dataset.filter(lambda example: (example["split"] == 'test'))
        else:
            raise ValueError('Not implied split option for huggingface pipeline.')

        def select_single_prompt(example, prompt_id):
            example["text"] = example["text"][prompt_id]
            return example


        tokenize_function = lambda x: graph_text_tokenizer_dict[args.transformer_backbone](examples=x,
                                                                                           tokenizer=tokenizer,
                                                                                           text_column_name='text',
                                                                                           padding=False,
                                                                                           max_seq_length=None,
                                                                                           rich_features=args.rich_features,
                                                                                           transform_in_collator=(
                                                                                               args.transform_in_collator))

        data_collator = graph_text_collator_dict[args.transformer_backbone](
            tokenizer=tokenizer,
            transform_in_collator=args.transform_in_collator,
            rich_features=args.rich_features)


    if not args.use_huggingface_pipeline: #Different pre-processing for the two types of pipelines.

        if args.task_policy =='traversal':
            recurrent_range=range(num_tasks)
        elif args.task_policy =='single':
            recurrent_range = [args.single_split]
        else:
            raise ValueError('prompt_policy not implemented yet')

        if args.not_retest_tasks_in_result_file:
            if os.path.exists(args.output_result_to_file):
                result_file=pd.read_csv(args.output_result_to_file,header=0,index_col=0)
            else:
                result_file=None

        for single_split_label in recurrent_range:
            if args.task_policy in ['traversal','single']:
                print('label split: ',single_split_label)
            if not str(single_split_label) in prompt:
                print('No prompt for label split {}'.format(single_split_label))
                continue
            
            if args.not_retest_tasks_in_result_file and result_file is not None:
                if len(result_file[(result_file['dataset']==args.dataset) & (result_file['split']==single_split_label)])>0:
                    print(args.dataset,' ',single_split_label,'has been tested')
                    continue
            

            train_loader.dataset.set_single_split(single_split_label)
            val_loader.dataset.set_single_split(single_split_label)
            test_loader.dataset.set_single_split(single_split_label)

            dataset.set_single_split(single_split_label)
            if args.comparison_protocol_dir:
                comparison_manifest, comparison_indices = comparison_protocol.apply_task(
                    args, single_split_label, smiles_list,
                    dataset.data.y[:, single_split_label].tolist(),
                    prompt[str(single_split_label)])
                train_index_total = comparison_indices['train']
                valid_index_total = comparison_indices['valid']
                test_index_total = comparison_indices['test']
                val_loader = DataLoader(dataset[valid_index_total], batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.num_workers, collate_fn=data_collator)
                test_loader = DataLoader(dataset[test_index_total], batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers, collate_fn=data_collator)
                val_loader.dataset.set_single_split(single_split_label)
                test_loader.dataset.set_single_split(single_split_label)
                print('Shared protocol:', comparison_manifest['protocol_sha256'])
            elif args.few_shot is not None:
                ind_each_class = {}
                for ind in train_index_total:
                    label=int(dataset[ind].y)
                    if label not in ind_each_class:
                        ind_each_class[label]=[ind]
                    else:
                        ind_each_class[label].append(ind)

                for key in ind_each_class.keys():
                    ind_each_class[key]=np.random.choice(ind_each_class[key], size=min(len(ind_each_class[key]),args.few_shot),replace=False).tolist()
                train_index_total=[]
                for key in ind_each_class.keys():
                    train_index_total+=ind_each_class[key]

            train_dataset = dataset[train_index_total]
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers, collate_fn=data_collator)
            train_loader.dataset.set_single_split(single_split_label)

            if args.export_few_shot_dir:
                from comparison_protocol import export_task, settings_from_args
                manifest_path = export_task(
                    args.export_few_shot_dir, args.dataset, single_split_label, smiles_list,
                    dataset.data.y[:, single_split_label].tolist(),
                    dict(train=train_index_total, valid=valid_index_total, test=test_index_total),
                    [prompt[str(single_split_label)][pid]
                     for pid in args.prompt_id[str(single_split_label)]], settings_from_args(args))
                print('Exported comparison protocol:', manifest_path)

            if args.prompt_policy == 'single':
                print(prompt[args.prompt_id[0]])

                #add prompt to graph data by data transform
                transform=lambda x: add_prompt_transform_dict[args.transformer_backbone](
                        data=x,data_label=x.y,input_ids=input_ids[args.prompt_id[0]],
                        attention_mask=attention_mask[args.prompt_id[0]],label_dict=label_dict,
                        rich_features=args.rich_features,transform_in_collator=args.transform_in_collator,
                    raw_prompts=prompt[args.prompt_id[0]],raw_label=raw_label,tokenizer=tokenizer,
                    generaltive_label=(task_type(args.dataset)=='reg'))

                train_loader.dataset.transform = transform
                val_loader.dataset.transform = transform
                test_loader.dataset.transform = transform

                downstream_task_by_transform(model,train_loader,val_loader,test_loader,prompt[args.prompt_id[0]])

            elif args.prompt_policy == 'traversal':
                for prompt_id in args.prompt_id[str(single_split_label)]:
                    print(prompt[str(single_split_label)][prompt_id])

                    transform=lambda x: add_prompt_transform_dict[args.transformer_backbone](
                        data=x,data_label=x.y,input_ids=input_ids[str(single_split_label)][prompt_id],
                        attention_mask=attention_mask[str(single_split_label)][prompt_id],label_dict=label_dict,
                        rich_features=args.rich_features,transform_in_collator=args.transform_in_collator,
                        raw_prompts=prompt[str(single_split_label)][prompt_id],raw_label=raw_label,tokenizer=tokenizer,
                        generaltive_label=(task_type(args.dataset)=='reg'))

                    train_loader.dataset.transform = transform
                    val_loader.dataset.transform = transform
                    test_loader.dataset.transform = transform

                    downstream_task_by_transform(model,train_loader,val_loader,test_loader,prompt[str(single_split_label)][prompt_id])

            else:
                raise ValueError('prompt_policy not implemented yet')

    else: #Huggingface pipelie

        if args.task_policy == 'traversal':
            recurrent_range = range(num_tasks)
        elif args.task_policy == 'single':
            recurrent_range = [args.single_split]
        else:
            raise ValueError('prompt_policy not implemented yet')

        if args.not_retest_tasks_in_result_file:
            if os.path.exists(args.output_result_to_file):
                result_file = pd.read_csv(args.output_result_to_file, header=0, index_col=0)
            else:
                result_file = None

        for single_split_label in recurrent_range:
            if args.task_policy in ['traversal', 'single']:
                print('label split: ', single_split_label)

            if args.not_retest_tasks_in_result_file and result_file is not None:
                if len(result_file[
                           (result_file['dataset'] == args.dataset) & (
                                   result_file['split'] == single_split_label)]) > 0:
                    print(args.dataset, ' ', single_split_label, 'has been tested')
                    continue

            train_dataset_task = train_dataset_total.filter(
                lambda example: (example["task_index"] == str(single_split_label)))
            valid_dataset_task = valid_dataset_total.filter(
                lambda example: (example["task_index"] == str(single_split_label)))
            test_dataset_task = test_dataset_total.filter(lambda example: (example["task_index"] == str(single_split_label)))

            if len(test_dataset_task) == 0:
                print('No label or prompt for label split {}'.format(single_split_label))
                continue

            if args.prompt_policy == 'single':
                # print()
                prompt_id_range = [args.prompt_id[0]]
            elif args.prompt_policy == 'traversal':
                if args.prompt_id is None:
                    prompt_id_range = range(len(train_dataset_task[0]['text']))
                else:
                    prompt_id_range = args.prompt_id[str(single_split_label)]
            else:
                raise ValueError('prompt_policy not implemented yet')

            for prompt_id in prompt_id_range:

                train_dataset = train_dataset_task.map(lambda example: select_single_prompt(example, prompt_id))
                valid_dataset = valid_dataset_task.map(lambda example: select_single_prompt(example, prompt_id))
                test_dataset = test_dataset_task.map(lambda example: select_single_prompt(example, prompt_id))

                prompt = train_dataset[0]['text']
                print(prompt)

                train_dataset = train_dataset.map(
                    tokenize_function,
                    batched=False,
                    num_proc=None,
                    remove_columns=['text'],
                    load_from_cache_file=not args.overwrite_data_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )
                valid_dataset = valid_dataset.map(
                    tokenize_function,
                    batched=False,
                    num_proc=None,
                    remove_columns=['text'],
                    load_from_cache_file=not args.overwrite_data_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )
                test_dataset = test_dataset.map(
                    tokenize_function,
                    batched=False,
                    num_proc=None,
                    remove_columns=['text'],
                    load_from_cache_file=not args.overwrite_data_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )

                train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers, collate_fn=data_collator)
                val_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.num_workers, collate_fn=data_collator)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers, collate_fn=data_collator)

                if args.few_shot is not None:
                    ind_each_class = {}
                    for ind, data in enumerate(train_dataset):
                        label = data['label']
                        if label not in ind_each_class:
                            ind_each_class[label] = [ind]
                        else:
                            ind_each_class[label].append(ind)

                    for key in ind_each_class.keys():
                        ind_each_class[key] = np.random.choice(ind_each_class[key],
                                                               size=min(len(ind_each_class[key]), args.few_shot),
                                                               replace=False).tolist()
                    train_index_total = []
                    for key in ind_each_class.keys():
                        train_index_total += ind_each_class[key]

                    train_dataset = train_dataset.select(train_index_total)
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers, collate_fn=data_collator)

                downstream_task_by_transform(model, train_loader, val_loader, test_loader,
                                             prompt)
