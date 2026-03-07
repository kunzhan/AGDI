GPU=0,1,2,3
TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
TRAINING_LOG=GPU_${GPU}_AGDI_${TRAINING_TIMESTAMP}
echo "$TRAINING_LOG"
mkdir ./log/$TRAINING_LOG

bash AGDI_llava.sh $GPU 2>&1 | tee ./log/$TRAINING_LOG/log.txt