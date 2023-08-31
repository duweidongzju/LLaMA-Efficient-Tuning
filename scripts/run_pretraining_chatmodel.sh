#!/bin/bash

model_name=qwen-7b-chat
path_to_llama_model=/home/username/MODEL/$model_name
path_to_pt_checkpoint=/home/username/OUTPUT/$model_name/pt

mkdir -p $path_to_pt_checkpoint

INF_NAN_MODE_ENABLE=1 python src/train_bash.py \
    --stage pt \
    --model_name_or_path $path_to_llama_model \
    --do_train \
    --dataset wiki_demo \
    --template chatml \
    --finetuning_type lora \
    --lora_target c_attn \
    --output_dir $path_to_pt_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --fp16 \
    --overwrite_output_dir
