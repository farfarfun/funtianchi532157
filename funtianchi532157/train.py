import os
import subprocess

from farlog import getLogger

logger = getLogger("funtianchi.train")


# data-juicer 代码所在目录
home_path = os.path.dirname(__file__)

logger.info("home_path: {}", home_path)
# 清洗后数据集的保存路径
trainer_path = os.path.join(home_path, "lm-training")
refine_data_en_path = f"{home_path}/outputs/refined_data/en_refine.jsonl"
refine_data_zh_path = f"{home_path}/outputs/refined_data/zh_refine.jsonl"
origin_model_path = f"{home_path}/data/models/falcon-rw-1b"
finetuned_model_path = f"{home_path}/outputs/finetuned_model"
eval_path = f"{home_path}/lm-evaluation-harness"
eval_result_path = f"{home_path}/outputs/eval_results"
challenge_data_path = f"{home_path}/data/challenge-data"


def step31() -> None:
    """清洗中英文训练数据。"""
    refine_data_cmd = [
        "python",
        f"{home_path}/process/process_data.py",
        "--config",
        f"{home_path}/configs/alpaca_cot/alpaca-cot-en-refine.yaml",
        "--dataset_path",
        f"{home_path}/data/raw_data/raw_data_en.jsonl",
        "--export_path",
        refine_data_en_path,
        "--limit",
        "20000",
        "--np",
        "4",
    ]
    subprocess.run(refine_data_cmd, check=True)
    refine_data_cmd = [
        "python",
        f"{home_path}/process/process_data.py",
        "--config",
        f"{home_path}/configs/alpaca_cot/alpaca-cot-zh-refine.yaml",
        "--dataset_path",
        f"{home_path}/data/raw_data/raw_data_zh.jsonl",
        "--export_path",
        refine_data_zh_path,
        "--limit",
        "20000",
        "--np",
        "4",
    ]
    subprocess.run(refine_data_cmd, check=True)


def step41() -> None:
    """使用清洗后的英文数据微调模型。"""
    training_cmd = [
        "bash",
        f"{trainer_path}/train_scripts/1b-T4/train.sh",
        origin_model_path,
        refine_data_en_path,
        finetuned_model_path,
    ]
    logger.info("training_cmd: {}", " ".join(training_cmd))
    subprocess.run(training_cmd, check=True)


os.environ["PYTHONPATH"] = eval_path


def step51() -> None:
    """评估微调后的模型。"""
    # 组装评估命令
    evaluation_cmd = [
        "bash",
        f"{eval_path}/examples/challenge-1B-stage1.sh",
        "dev",
        finetuned_model_path,
        challenge_data_path,
        eval_result_path,
    ]
    subprocess.run(evaluation_cmd, cwd=home_path, check=True)


if __name__ == "__main__":
    step41()
