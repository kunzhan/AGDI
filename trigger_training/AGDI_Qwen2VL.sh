echo $1
export CUDA_VISIBLE_DEVICES=$1
# python AGDI_Qwen2VL.py \
#     --batch_size 1 \
#     --num_samples 200 \
#     --input_res 224\
#     --epsilon 16\
#     --steps 1000 \
#     --alpha 1.0\
#     --beta 0.001\
#     --eps1 0.0005\
#     --lamb 1.0 \
#     --model_path "PATH"\
#     --data_json_path "PATH/special_merge1000.json"\
#     --output "PATH" \
#     --cle_data_path  "PATH/imagenet/random_img/"\



python AGDI_Qwen2VL.py \
    --batch_size 1 \
    --num_samples 200 \
    --input_res 224\
    --epsilon 16\
    --steps 1000 \
    --alpha 1.0\
    --beta 0.001\
    --eps1 0.0005\
    --lamb 1.0 \
    --model_path "/data/xcw/model/Qwen2-VL-2B-Instruct/"\
    --data_json_path "/data/xcw/dataset/PLA_prune/mydataset/special_merge1000.json"\
    --output "/data/xcw/dataset/PLA_prune/5pair_combine100/PLA/ADGI_Qwen2VL_adaptive_rate" \
    --cle_data_path  "/data/xcw/dataset/imagenet/random_img/"\



