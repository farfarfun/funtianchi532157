# funtianchi532157

用于数据清洗、语言模型训练和评估的实验脚本集合。训练和评估需要相应的
GPU、模型权重及数据集，默认路径见 `funtianchi/train.py`。

## 安装

需要 Python 3.10 及 uv：

```bash
uv sync
```

准备好数据和模型后，可运行完整流程：

```bash
./funtianchi/process/steps.sh step31
python funtianchi/train.py
```

`funtianchi/process/process_data.py` 负责数据处理，`funtianchi/lm-training/`
包含训练脚本，`funtianchi/lm-evaluation-harness/` 包含评估入口。各步骤失败
时命令会返回非零状态。

## 第三方代码

`funtianchi/lm-evaluation-harness/` 基于
[EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)，
当前保留版本 `0.3.0`，原始代码按 Apache License 2.0 发布；其版权和许可证
文件保留在该目录中。

## 关于 farfarfun

farfarfun 是一个开放的组织，致力于分享实用的开源工具和实验项目。
项目统一采用 [MIT License](LICENSE)。
