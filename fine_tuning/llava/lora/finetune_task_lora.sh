#!/bin/bash


# visual7w
deepspeed --include localhost:1,2,3,4 --master_port=25641 YOUR_PATH/LLaVA/train_mem.py \
    --lora_enable True --lora_r 16 --lora_alpha 32  \
    --deepspeed YOUR_PATH/LLaVA/scripts/zero3.json \
    --model_name_or_path YOUR_PATH/llava-v1.5-7b \
    --version v1 \
    --data_path YOUR_PATH/train28k.json \
    --image_folder YOUR_PATH/visual7w/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir YOUR_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

deepspeed --include localhost:1,2,3,4 --master_port=25641 YOUR_PATH/LLaVA/train_mem.py \
    --lora_enable True --lora_r 16 --lora_alpha 32  \
    --deepspeed YOUR_PATH/LLaVA/scripts/zero3.json \
    --model_name_or_path YOUR_PATH/llava-v1.5-7b \
    --version v1 \
    --data_path YOUR_PATH/TextVQA/sharegpt_train.json \
    --image_folder YOUR_PATH/TextVQA/train_images/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir YOUR_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

# # # ST-VQA
deepspeed --include localhost:1,2,3,4 --master_port=25641 YOUR_PATH/LLaVA/train_mem.py \
    --lora_enable True --lora_r 16 --lora_alpha 32  \
    --deepspeed YOUR_PATH/LLaVA/scripts/zero3.json \
    --model_name_or_path YOUR_PATH/llava-v1.5-7b \
    --version v1 \
    --data_path YOUR_PATH/ST-VQA/sharegpt_train.json \
    --image_folder YOUR_PATH/ST-VQA/train_data/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir YOUR_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

# MathV360K
deepspeed --include localhost:4,5,6,7 --master_port=25641 YOUR_PATH/LLaVA/train_mem.py \
    --lora_enable True --lora_r 16 --lora_alpha 32  \
    --deepspeed YOUR_PATH/LLaVA/scripts/zero3.json \
    --model_name_or_path YOUR_PATH/llava-v1.5-7b \
    --version v1 \
    --data_path YOUR_PATH/MathV360K/train50k_filt.json \
    --image_folder YOUR_PATH/MathV360K/data_images/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir YOUR_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    # --report_to wandb


# # PaintingForm
deepspeed --include localhost:1,2,3,4 --master_port=25641 YOUR_PATH/LLaVA/train_mem.py \
    --lora_enable True --lora_r 16 --lora_alpha 32 \
    --deepspeed YOUR_PATH/LLaVA/scripts/zero3.json \
    --model_name_or_path YOUR_PATH/llava-v1.5-7b \
    --version v1 \
    --data_path YOUR_PATH/PaintingForm/train20k_filt.json \
    --image_folder YOUR_PATH/PaintingForm/art_images_data/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir YOUR_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
















































