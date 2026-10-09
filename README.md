# funtianchi532157

阿里天池 [FT-Data Ranker](https://tianchi.aliyun.com/competition/entrance/532157)
大模型微调数据竞赛（1B 模型赛道）参赛代码：基于 Data-Juicer 的中英文数据清洗配置、
falcon-rw-1b 微调脚本，以及 lm-evaluation-harness 评测流水线。训练和评估需要相应的
GPU、模型权重及数据集，默认路径见 `funtianchi532157/train.py`。

## 安装

需要 Python 3.12（由依赖 `funget` 的下限决定）及 uv：

```bash
./funtianchi532157/install.sh
```

## 完整流程

`funtianchi532157/process/steps.sh` 按顺序定义了比赛的完整流程，各步骤失败时
命令会返回非零状态：

| 步骤 | 命令 | 说明 |
| --- | --- | --- |
| step11 | `./funtianchi532157/process/steps.sh step11` | 执行 `install.sh`，安装 lm-evaluation-harness 等子项目依赖 |
| step21 | `./funtianchi532157/process/steps.sh step21` | 执行 `prepare_data_and_models.sh`，通过 `funget` 下载评测数据、基础模型权重和原始训练数据到 `funtianchi532157/data/` |
| step31 | `./funtianchi532157/process/steps.sh step31` | 用 Data-Juicer 清洗中英文原始数据，产出 `funtianchi532157/outputs/refined_data/{en,zh}_refine.jsonl` |
| step41 | `./funtianchi532157/process/steps.sh step41` | 微调基础模型（需要 GPU），产出 `funtianchi532157/outputs/finetuned_model` |
| step51 | `./funtianchi532157/process/steps.sh step51` | 用 lm-evaluation-harness 在验证集上评估微调后的模型，产出 `funtianchi532157/outputs/eval_dev_results` |
| step61 | `./funtianchi532157/process/steps.sh step61` | 生成提交榜单用的评测结果，产出 `funtianchi532157/outputs/eval_board_results` |

数据和模型准备好后，可按序运行各步骤，例如：

```bash
./funtianchi532157/process/steps.sh step11
./funtianchi532157/process/steps.sh step21
./funtianchi532157/process/steps.sh step31
./funtianchi532157/process/steps.sh step41
./funtianchi532157/process/steps.sh step51
```

`funtianchi532157/train.py` 提供了 step31/step41/step51 的 Python 入口
（`python funtianchi532157/train.py` 默认执行 step41 微调）；
`funtianchi532157/process/process_data.py` 负责单次数据清洗调用，
`funtianchi532157/lm-training/` 包含训练脚本，`funtianchi532157/lm-evaluation-harness/`
包含评估入口。

## 第三方代码

- `funtianchi532157/lm-evaluation-harness/` 基于
  [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)，
  当前保留版本 `0.3.0`，原始代码按 Apache License 2.0 发布；其版权和许可证
  文件保留在该目录中。
- `funtianchi532157/lm-training/train.py` 基于
  [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) 的
  `train.py` 改写，原始代码按 Apache License 2.0 发布（版权归 Rohan Taori、
  Ishaan Gulrajani、Tianyi Zhang、Yann Dubois、Xuechen Li 所有）；本次改写
  替换为 farlog 记录日志并补充中文 docstring，完整许可证文本见同目录的
  `train.py.LICENSE`。

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
