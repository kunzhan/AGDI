# #!/bin/bash

echo $1
export CUDA_VISIBLE_DEVICES=$1




# # # visual7w
LLAVA_MODEL_PATH="YOUR_PATH/llava-v1.5-7b-task-lora-visual7w/" # Path to LLaMA pretrained model

python copyright_TMR_llava.py \
    --batch_size 1 \
    --num_samples 1000 \
    --img_path 'YOUR_PATH' \
    --output_path 'YOUR_PATH' \
    --llava_model_path "$LLAVA_MODEL_PATH" 




# # # TextVQA
LLAVA_MODEL_PATH="YOUR_PATH/llava-v1.5-7b-task-lora-TextVQA/" # Path to LLaMA pretrained model


python copyright_TMR_llava.py \
    --batch_size 1 \
    --num_samples 1000 \
    --img_path 'YOUR_PATH' \
    --output_path 'YOUR_PATH' \
    --llava_model_path "$LLAVA_MODEL_PATH" 






# # ST-VQA
LLAVA_MODEL_PATH="YOUR_PATH/llava_lora_STVQA/" # Path to LLaMA pretrained model


python copyright_TMR_llava.py \
    --batch_size 1 \
    --num_samples 1000 \
    --img_path 'YOUR_PATH' \
    --output_path 'YOUR_PATH' \
    --llava_model_path "$LLAVA_MODEL_PATH" 






# # MathV360K
LLAVA_MODEL_PATH="YOUR_PATH/llava-v1.5-7b-task-lora-MathV360K/" # Path to LLaMA pretrained model


python copyright_TMR_llava.py \
    --batch_size 1 \
    --num_samples 1000 \
    --img_path 'YOUR_PATH' \
    --output_path 'YOUR_PATH' \
    --llava_model_path "$LLAVA_MODEL_PATH" 







# PaintingForm
LLAVA_MODEL_PATH="YOUR_PATH/llava-v1.5-7b-task-lora-PaintingForm/" # Path to LLaMA pretrained model


python copyright_TMR_llava.py \
    --batch_size 1 \
    --num_samples 1000 \
    --img_path 'YOUR_PATH' \
    --output_path 'YOUR_PATH' \
    --llava_model_path "$LLAVA_MODEL_PATH" 
    






