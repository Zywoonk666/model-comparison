#!/bin/bash

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

if [ ! -d cache ]; then
  mkdir cache
fi

run_ablation() {
    local device=$1
    local exp_name=$2
    local use_projector=$3 
    local use_ot=$4            
    local use_progressive_mask=$5 
    local use_graph_expert=$6   
    local proj_dropout=$7      
    local epochs=$8
    local proj_blocks=${9:-2}
    local proj_act=${10:-relu} 
    local few_shot_number=128 
    local out_file_prefix="cache/"
    local outfile="${out_file_prefix}${exp_name}.csv"
    local dataset_list=(bace hiv bbbp tox21 esol freesolv)
    local comparison_args=()
    if [ -n "${EXPORT_FEW_SHOT_DIR:-}" ]; then
        comparison_args+=(--export_few_shot_dir "$EXPORT_FEW_SHOT_DIR")
    fi
    
    for dataset in "${dataset_list[@]}"; do
        CUDA_VISIBLE_DEVICES=$device python downstream_test.py \
            --use_projector $use_projector \
            --use_ot_alignment $use_ot \
            --use_progressive_mask $use_progressive_mask \
            --use_graph_expert $use_graph_expert \
            --projector_dropout $proj_dropout \
            --projector_blocks $proj_blocks \
            --projector_act $proj_act \
            --few_shot $few_shot_number \
            --few_shot_fashion innovation \
            --transformer_backbone gimlet \
            --model_name_or_path "haitengzhao/gimlet" \
            --tokenizer_name t5-small \
            --dataset "$dataset" \
            --runseed 6 \
            --batch_size 128 \
            --grad_accum_step 8 \
            --lr 1e-2 \
            --epochs $epochs \
            --test_interval 20 \
            --transform_in_collator \
            --no_eval_train \
            --output_result_to_file $outfile \
            "${comparison_args[@]}" \
            --not_retest_tasks_in_result_file
    done
}

DROP=0.5
EP=60
PREFIX="Drop${DROP}_Ep${EP}"

run_ablation 0 "${PREFIX}_Exp8_Full_Model" True True True True $DROP $EP
