# 更新日志

## 未发布

### 变更

- 增加 uv 项目元数据和锁文件管理。
- 补充安装、运行、第三方来源和组织说明。

### 修复

- 训练和数据处理子命令现在会在失败时立即退出。
- Shell 流程现在校验步骤参数并传播子命令错误。
- `train.py`（2 处）、`lm-training/train.py`（1 处）、
  `lm-training/get_train_dataset_1b.py`（2 处）、
  `lm-training/get_train_dataset_7b.py`（2 处）共 7 处日志误用 stdlib logging 的
  `%s` 占位符，farlog（loguru）不支持该语法，参数被静默丢弃；统一改为 `{}` 占位符。
