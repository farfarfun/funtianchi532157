#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "用法: $0 step11|step21|step31|step41|step51|step61" >&2
  exit 2
fi

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
model_path="$project_dir/data/models/falcon-rw-1b"
refine_data_path="$project_dir/outputs/refined_data"
finetuned_model_path="$project_dir/outputs/finetuned_model"
eval_dev_result_path="$project_dir/outputs/eval_dev_results"
eval_board_result_path="$project_dir/outputs/eval_board_results"

if [[ ${1} = "step11" ]]; then
  sh "$project_dir/install.sh"
elif [[ ${1} = "step21" ]]; then
  sh "$project_dir/prepare_data_and_models.sh"
elif [[ ${1} = "step31" ]]; then
  # 处理
  python "$project_dir/process/process_data.py" --config "$project_dir/configs/alpaca_cot/alpaca-cot-en-refine.yaml" --dataset_path "$project_dir/data/raw_data/raw_data_en.jsonl" --export_path "$refine_data_path/en_refine.jsonl" --np 10
  python "$project_dir/process/process_data.py" --config "$project_dir/configs/alpaca_cot/alpaca-cot-zh-refine.yaml" --dataset_path "$project_dir/data/raw_data/raw_data_zh.jsonl" --export_path "$refine_data_path/zh_refine.jsonl" --np 10
elif [[ ${1} = "step41" ]]; then
  # 训练
  bash "$project_dir/lm-training/train_scripts/1b-T4/train.sh" "$model_path" "$refine_data_path/en_refine.jsonl" "$finetuned_model_path"
elif [[ ${1} = "step51" ]]; then
  # 评估
  bash "$project_dir/lm-evaluation-harness/examples/challenge-1B-stage1.sh" dev "$finetuned_model_path" "$project_dir/data/challenge-data" "$eval_dev_result_path"
elif [[ ${1} = "step61" ]]; then
  # 提交
  bash "$project_dir/lm-evaluation-harness/examples/challenge-1B-stage1.sh" board "$finetuned_model_path" "$project_dir/data/challenge-data" "$eval_board_result_path"
else
  echo "不支持的步骤: ${1}" >&2
  echo "用法: $0 step11|step21|step31|step41|step51|step61" >&2
  exit 2
fi
