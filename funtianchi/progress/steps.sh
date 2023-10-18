#!/bin/bash

model_path='data/models/falcon-rw-1b'
refine_data_path='outputs/refined_data'
finetuned_model_path='outputs/finetuned_model'
eval_dev_result_path='outputs/eval_dev_results'
eval_board_result_path='outputs/eval_board_results'

echo $0
if [[ ${1} = "step11" ]]; then
  sh install.sh
elif [[ ${1} = "step21" ]]; then
  sh prepare_data_and_models.sh
elif [[ ${1} = "step31" ]]; then
  # 处理
  python data-juicer/tools/process_data.py --config data-juicer/configs/data_juicer_recipes/alpaca_cot/alpaca-cot-en-refine.yaml \
    --dataset_path data/raw_data/raw_data_en.jsonl --export_path $refine_data_path/en_refine.jsonl --np 10
  python data-juicer/tools/process_data.py --config data-juicer/configs/data_juicer_recipes/alpaca_cot/alpaca-cot-zh-refine.yaml \
    --dataset_path data/raw_data/raw_data_zh.jsonl --export_path $refine_data_path/zh_refine.jsonl --np 10
elif [[ ${1} = "step41" ]]; then
  # 训练
  sh lm-training/train_scripts/deepspeed_train_1b.sh $model_path $refine_data_path $finetuned_model_path
elif [[ ${1} = "step51" ]]; then
  # 评估
  sh lm-evaluation-harness/examples/challenge-1B-stage1.sh dev finetuned_model_path 'data/challenge-data' $eval_dev_result_path
elif [[ ${1} = "step61" ]]; then
  # 提交
  sh lm-evaluation-harness/examples/challenge-1B-stage1.sh board finetuned_model_path 'data/challenge-data' $eval_board_result_path
fi
