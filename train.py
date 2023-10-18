import os
import subprocess

# get data-juicer code directory
home_path = os.getcwd()
#home_path = '/home/admin'
home_path = os.path.join(home_path, 'funtianchi532157/funtianchi')

print(f"home_path: {home_path}")
# path to save refined dataset
trainer_path = os.path.join(home_path, 'lm-training')
refine_data_en_path = f'{home_path}/outputs/refined_data/en_refine.jsonl'
refine_data_zh_path = f'{home_path}/outputs/refined_data/zh_refine.jsonl'
origin_model_path = f'{home_path}/data/models/falcon-rw-1b'
finetuned_model_path = f'{home_path}/outputs/finetuned_model'
eval_path = f'{home_path}/lm-evaluation-harness'
eval_result_path = f'{home_path}/outputs/eval_results'


# os.environ['PYTHONPATH'] = dj_path


def step31():
    refine_data_cmd = [
        'python',
        f'{home_path}/process/process_data.py',
        '--config', f'{home_path}/configs/alpaca_cot/alpaca-cot-en-refine.yaml',
        '--dataset_path', f'{home_path}/data/raw_data/raw_data_en.jsonl',
        '--export_path', refine_data_en_path,
        '--limit','20000',
        '--np', '4']
    subprocess.run(refine_data_cmd)
    refine_data_cmd = [
        'python',
        f'{home_path}/process/process_data.py',
        '--config', f'{home_path}/configs/alpaca_cot/alpaca-cot-en-refine.yaml',
        '--dataset_path', f'{home_path}/data/raw_data/raw_data_zh.jsonl',
        '--export_path', refine_data_zh_path,
        '--limit','20000',
        '--np', '4']
    subprocess.run(refine_data_cmd)


def step41():
    training_cmd = ['bash',
                    f'{trainer_path}/train_scripts/train.sh',
                    origin_model_path,
                    refine_data_en_path,
                    finetuned_model_path]
    print(f'training_cmd: {" ".join(training_cmd)}')
    subprocess.run(training_cmd)
    

os.environ['PYTHONPATH'] = eval_path


def step51():
    # configure data processing command
    evaluation_cmd = ['bash',
                      f'{eval_path}/examples/challenge-1B-stage1.sh',
                      'dev',
                      finetuned_model_path,
                      '/srv/data/challenge-data', eval_result_path]
    subprocess.run(evaluation_cmd, cwd=home_path)

step41()