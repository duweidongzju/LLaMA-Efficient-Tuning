#!/bin/bash

model_name=llama-2-7b-hf
path_to_llama_model=/home/username/MODEL/$model_name
path_to_pt_checkpoint=/home/username/CHECKPOINT/$model_name/pt
mkdir -p $path_to_pt_checkpoint

INF_NAN_MODE_ENABLE=1 accelerate launch scr/train_bash.py \
    --stage pt \
    --model_name_or_path $path_to_llama_model \
    --do_train \
    --dataset wiki_demo \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir $path_to_pt_checkpoint \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 10.0 \
    --fp16
