#!/bin/bash
echo "train begin"
set -e 
if [ -z $XDG_CACHE_HOME ]; then
    export XDG_CACHE_HOME=$HOME/.cache
fi

if [[ $# -ne 3 ]]; then
    echo "Three arguments required! " >&2
    exit 2
fi

# Model Path
# e.g /home/model/baichuan2-7b/
model_path=${1} #/path/to/your/model/
tokenizer=${model_path}

# Data Path
# e.g /home/data/train.jsonl
data_path=${2} # /path/to/your/dataset.jsonl

# Output Path
# e.g ${WORK_DIR}/checkpoints/baichuan2-7b/
output_path=${3} #/path/to/your/output/

mkdir -p ${output_path}/

WORK_DIR=$(echo `cd $(dirname $0); pwd | xargs dirname`)
cd ${WORK_DIR}

# Train Parameter
bs_per_gpu=1
grad_acc=32

python train.py \
    --model_name_or_path ${model_path} \
    --tokenizer ${tokenizer} \
    --data_path ${data_path} \
    --output_dir ${output_path} \
    --per_device_train_batch_size ${bs_per_gpu} \
    --gradient_accumulation_steps ${grad_acc} \
    --lang en \
    --gradient_checkpointing_enable True \
    --num_train_epochs 1 \
    --model_max_length 1024 \
    --learning_rate 2.5e-5 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps -1 \
    --save_total_limit 999 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False | tee ${output_path}/training_log.txt

echo "train done"