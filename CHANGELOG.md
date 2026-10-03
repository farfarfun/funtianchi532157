# 更新日志

## 未发布

### 变更

- 增加 uv 项目元数据和锁文件管理。
- 补充安装、运行、第三方来源和组织说明。
- `requires-python` 由 `>=3.10` 提升到 `>=3.12`，以满足新声明依赖 `funget` 的
  Python 版本下限；`lm-evaluation-harness` 子项目的 `requires-python` 同步从
  `>=3.9` 提升到组织基线 `>=3.10`。
- `prepare_data_and_models.sh` 使用的下载工具 `funget` 补充声明到
  `pyproject.toml` 的 `dependencies`。
- README 补充 step11/21/31/41/51/61 各步骤的用途、产出路径和 GPU 前置条件，
  以及 `lm-training/train.py` 的第三方来源（Stanford Alpaca，Apache License 2.0）。
- `lm-training/train.py` 的 `print_rank`、`smart_tokenizer_and_embedding_resize`、
  `preprocess`、`format_data`、`update_token_id`、`train` 补充中文 docstring。

### 修复

- 训练和数据处理子命令现在会在失败时立即退出。
- Shell 流程现在校验步骤参数并传播子命令错误。
- `train.py`（2 处）、`lm-training/train.py`（1 处）、
  `lm-training/get_train_dataset_1b.py`（2 处）、
  `lm-training/get_train_dataset_7b.py`（2 处）共 7 处日志误用 stdlib logging 的
  `%s` 占位符，farlog（loguru）不支持该语法，参数被静默丢弃；统一改为 `{}` 占位符。
- `train.py` 的 `step31()` 清洗中文数据时错误地复用了英文 refine 配置
  （`alpaca-cot-en-refine.yaml`），导致中文数据从未按中文配置清洗；改为
  `alpaca-cot-zh-refine.yaml`。
- `train.py` 的 `step51()` 评估路径硬编码为 `/srv/data/challenge-data`，
  在非原作者机器上必定找不到数据；改为基于仓库自身位置计算的
  `{home_path}/data/challenge-data`，与 `steps.sh` 保持一致。
