python data-juicer/tools/process_data.py --config data-juicer/configs/data_juicer_recipes/alpaca_cot/alpaca-cot-en-refine.yaml --dataset_path data/raw_data/raw_data_en.jsonl --export_path outputs/refined_data/en_refine.jsonl --np 4
python data-juicer/tools/process_data.py --config data-juicer/configs/data_juicer_recipes/alpaca_cot/alpaca-cot-zh-refine.yaml --dataset_path data/raw_data/raw_data_zh.jsonl --export_path outputs/refined_data/zh_refine.jsonl --np 4

