#!/bin/bash
dataset="mimic_cxr"
base_dir="./mimic-cxr-jpg/2.0.0/files/"
sn_annotation="./final_single_view_no_long_add1score_sentence_level.json"
sw_annotation="./final_single_view_with_long_add1score_sentence_level.json"
mn_annotation="./final_multi_view_no_long_add1score_sentence_level.json"
mw_annotation="./final_multi_view_with_long_add1score_sentence_level.json"
vicuna_model="./hf/vicuna-7b-v1.5"
rad_dino_path="./hf/rad-dino"
cxr_bert_path="./hf/BiomedVLP-CXR-BERT-specialized"
chexbert_path="./hf/chexbert.pth"
bert_path="./hf/bert-base-uncased"
version="train_stage2"
stage1_ckpt_path="/private/workspace/yangx/data/mm/Real/save/mimic_cxr/vicuna_sentencelevel_1_75_128token_e-4_direct_stage1_fix/pths/checkpoint_epoch1_step14384_bleu0.165496_cider0.280208_chexbert0.381323.pth"
srage2_ckpt_path="/private/workspace/yangx/data/mm/Real/save/mimic_cxr/final_iit_withloss_A100/pths/checkpoint_epoch1_step43152_bleu0.249316_cider0.745028_chexbert0.571643.pth"
test_mode="sn"
savepath="./save/$dataset/$version/$test_mode"
if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi
python -u train.py \
    --test \
    --dataset ${dataset} \
    --sn_annotation ${sn_annotation} \
    --sw_annotation ${sw_annotation} \
    --mn_annotation ${mn_annotation} \
    --mw_annotation ${mw_annotation} \
    --base_dir ${base_dir} \
    --vicuna_model ${vicuna_model} \
    --rad_dino_path ${rad_dino_path} \
    --cxr_bert_path ${cxr_bert_path} \
    --chexbert_path ${chexbert_path} \
    --bert_path ${bert_path} \
    --batch_size 16 \
    --val_batch_size 4 \
    --freeze_vm True \
    --savedmodel_path ${savepath} \
    --max_length 100 \
    --min_new_tokens 50 \
    --max_new_tokens 200 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 12 \
    --devices 1 \
    --max_epochs 2 \
    --limit_val_batches 0.5 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 2 \
    --stage_class 2 \
    --llm_use_lora True \
    --llm_r 32 \
    --llm_alpha 64 \
    --lora_dropout 0.1 \
    --accumulate_grad_batches 2 \
    --loss_mode 'sentence' \
    --sentence_ratio 0.75 \
    --learning_rate 3e-4 \
    --visual_token_number 128 \
    --test_mode ${test_mode} \
    --test_batch_size 8 \
    --delta_file ${srage2_ckpt_path} \
    2>&1 | tee -a ${savepath}/log.txt
