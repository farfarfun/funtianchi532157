#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#    说明：本文件基于 Stanford Alpaca（tatsu-lab/stanford_alpaca）的
#    train.py 改写而来，上游以 Apache License 2.0 发布；本次修改包括改用
#    farlog 记录日志、补充中文 docstring 等，完整许可证文本见同目录
#    train.py.LICENSE。
import copy
import logging
import warnings
import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import transformers
from datasets import load_dataset, Dataset
from transformers import Trainer
from tqdm import tqdm
from farlog import getLogger

logger = getLogger("funtianchi.lm-training")

warnings.filterwarnings("ignore")


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "en": {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )},
    "zh": {
    "prompt_input": (
        "以下是描述任务的指示，配有提供进一步上下文的输入，编写一个适当的回应完成请求\n\n"
        "### 指示：\n{instruction}\n\n### 输入：\n{input}\n\n### 回应："
    ),
    "prompt_no_input": (
        "以下是描述任务的指示，编写一个适当的回应完成请求\n\n"
        "### 指示：\n{instruction}\n\n### 回应："
    )}
}


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(default="")
    tokenizer: str = field(default=None)
    gradient_checkpointing_enable: bool=field(default=True)


@dataclass
class DataArguments:
    data_path: list[str]
    lang: str = field(default="en")
    num_proc: int = field(default=1)
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str | None = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    only_update_ffn: bool=field(default=False)
    only_update_attn: bool=field(default=False)
    
    enable_lora: bool=field(default=False)
    target_modules: list[str] | None = field(
        default=None, metadata={"help": "Name(s) of target modules to apply LoRA"}
    )
    lora_r: int=field(default=8)
    lora_alpha: int=field(default=32)
    lora_dropout: float=field(default=0.1)
    report_to: str=field(default="none")


def print_rank(*args, **kwargs):
    """仅在分布式训练的 rank 0 进程输出日志，避免多进程重复打印。

    参数:
        *args: 待输出的内容，按空格拼接后记录。
        **kwargs: 预留参数，当前未使用。
    """
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    if local_rank == 0:
        logger.info("{}", " ".join(map(str, args)))


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict[str, str],
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """调整 tokenizer 和模型 embedding 的大小以容纳新增的特殊 token。

    注意：这是未做显存对齐优化的版本，调整后的 embedding 维度可能不是 64 的倍数。

    参数:
        special_tokens_dict: 待新增的特殊 token，键为 token 类型
            （如 `pad_token`），值为 token 文本。
        tokenizer: 需要新增特殊 token 的分词器，原地修改。
        model: 需要同步调整 embedding 大小的模型，原地修改。

    返回:
        None，所有修改均原地作用于传入的 `tokenizer` 和 `model`。
    """
    def _get_resized_lm_head(
        self, old_lm_head, new_num_tokens: int | None = None, transposed: bool = False
    ) :
        """
        Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_lm_head (`torch.nn.Linear`):
                Old lm head liner layer to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Linear` module of the model without doing anything. transposed (`bool`, *optional*, defaults
                to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.size()` is `lm_head_dim,
                vocab_size` else `vocab_size, lm_head_dim`.

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is
            `None`
        """

        if new_num_tokens is None:
            return old_lm_head
        from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_lm_head.weight, modifier_rank=None):
                old_num_tokens, old_lm_head_dim = (
                    old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
                )
        else:
            old_num_tokens, old_lm_head_dim = (
                old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
            )

        if old_num_tokens == new_num_tokens:
            return old_lm_head

        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = False
        if hasattr(old_lm_head, 'bias'):
            has_new_lm_head_bias = old_lm_head.bias is not None

        new_lm_head = old_lm_head.__class__(*new_lm_head_shape, bias=has_new_lm_head_bias)
        new_lm_head.weight = torch.nn.Parameter(new_lm_head.weight.to(
                                                    device=old_lm_head.weight.device,
                                                    dtype=old_lm_head.weight.dtype))

        # initialize new lm head (in particular added tokens)
        self._init_weights(new_lm_head)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        # XXX: put the long block of code in a wrapper
        
        if is_deepspeed_zero3_enabled():
            import deepspeed
            if has_new_lm_head_bias:
                params = [old_lm_head.weight, old_lm_head.bias, new_lm_head.weight, new_lm_head.bias]
            else:
                params = [old_lm_head.weight, new_lm_head.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    # Copy old lm head weights to new lm head
                    if not transposed:
                        new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[
                            :num_tokens_to_copy, :
                        ]
                    else:
                        new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[
                            :, :num_tokens_to_copy
                        ]

                    # Copy bias weights to new lm head
                    if has_new_lm_head_bias:
                        new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]
        else:
            # Copy old lm head weights to new lm head
            if not transposed:
                new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
            else:
                new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

            # Copy bias weights to new lm head
            if has_new_lm_head_bias:
                new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

        return new_lm_head
    
    
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    old_lm_head = model.get_output_embeddings()
    from torch import nn
    if old_lm_head is not None and not isinstance(old_lm_head, nn.Linear):
        #setattr(model, '_get_resized_lm_head',  _get_resized_other_head)
        from types import MethodType
        
        model._get_resized_lm_head = MethodType(_get_resized_lm_head, model)

    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0 :
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def preprocess(
    format_dataset: Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    num_proc:int = 1,
) -> dict[str, Any]:
    """对格式化后的数据做分词，生成训练所需的 input_ids/labels。

    参数:
        format_dataset: `format_data` 产出的、已拼好 source/target 文本的数据集。
        tokenizer: 用于分词的分词器。
        num_proc: 并行处理的进程数。

    返回:
        分词、过滤超长样本后的数据集，列包含 `input_ids`、`labels`。
    """

    def _tokenize_fn(example):
        """Tokenize example"""
        example["source"] = tokenizer(example["source"], return_tensors="pt", padding="longest",
                                      max_length=tokenizer.model_max_length, truncation=True,
                                      add_special_tokens=False)
        example["target"] = tokenizer(example["target"], return_tensors="pt", padding="longest",
                                      max_length=tokenizer.model_max_length, truncation=True,
                                      add_special_tokens=False)
                            
        source_input_id = source_label = example["source"].input_ids[0]          
        target_input_id = target_label = torch.cat([example["target"].input_ids[0], torch.tensor([tokenizer.eos_token_id])])
        input_id = torch.cat([source_input_id, target_input_id])
        label = copy.deepcopy(input_id)
        label[:len(source_input_id)] = IGNORE_INDEX

        example["input_ids"] = input_id
        example["labels"] = label
        example["split_ids"] = len(source_input_id)
        return example

    processed_dataset = format_dataset.map(_tokenize_fn, remove_columns=["source", "target"],num_proc=num_proc)
    processed_dataset = processed_dataset.filter(lambda x: len(x["input_ids"]) <= tokenizer.model_max_length, num_proc=num_proc)
    processed_dataset.set_format("pt", columns=["input_ids", "labels"], output_all_columns=True)

    id_choice = random.sample(range(len(processed_dataset)), min(5, len(processed_dataset)))
    for i in id_choice:
        split_id = processed_dataset[i]["split_ids"]
        print_rank("PROMPT:")
        print_rank(repr(tokenizer.decode(processed_dataset[i]["input_ids"][:split_id])))
        print_rank("RESPONSE:")
        print_rank(repr(tokenizer.decode(processed_dataset[i]["labels"][split_id:])))
        print_rank("=" * 100)
    print_rank(f"ORI NUMBER: {len(format_dataset)}, AFTER FILETER: {len(processed_dataset)}, DROP NUMBER: {len(format_dataset) - len(processed_dataset)}")
    return processed_dataset


def format_data(lang: str, dataset: Dataset, num_proc:int = 1):
    """按语言选用对应的 prompt 模板，拼接出训练所需的 source/target 字段。

    参数:
        lang: 数据语言，取值 `en` 或 `zh`，决定使用的 prompt 模板。
        dataset: 原始数据集，每条样本须包含 `instruction`、`output`，
            `input` 可选。
        num_proc: 并行处理的进程数。

    返回:
        新数据集，`instruction`/`input`/`output` 列被替换为
        `source`/`target` 两列；若样本缺少 `instruction` 或 `output`
        字段，抛出 `RuntimeError`。
    """
    prompt_input, prompt_no_input = PROMPT_DICT[lang]["prompt_input"], PROMPT_DICT[lang]["prompt_no_input"]
    def add_prompt(example):
        if "instruction" in example and "output" in example:
            example["target"] = example["output"]
            if example.get("input", "") != "":
                example["source"] = prompt_input.format_map(example)
            else:
                example["source"] = prompt_no_input.format_map(example)
            return example
        else:
            raise RuntimeError(f"{example}")
    return dataset.map(add_prompt, remove_columns=["instruction", "input", "output"], num_proc=num_proc)
    

class SupervisedDataset(torch.utils.data.Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        lang, data_path, num_proc= data_args.lang, data_args.data_path, data_args.num_proc

        dataset = load_dataset('json', data_files=data_path, split="train")
        print_rank(f"There are {len(dataset)} training samples in data path")

        print_rank("Formatting inputs...")
        # * add different formats
        format_dataset = format_data(lang, dataset,num_proc = num_proc)
        
        print_rank("Tokenizing inputs... This may take some time...")
        self.dataset  = preprocess(format_dataset, tokenizer, num_proc = num_proc)
        num_tokens = sum(map(lambda x : len(x['input_ids']), self.dataset))
        print_rank(f"Total {len(self.dataset)} samples [{num_tokens/10**6: .2f}M tokens] in training!")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return self.dataset[i]


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> dict[str, Any]:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(data_args, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def update_token_id(model, tokenizer):
    """修正 tokenizer 与模型配置中特殊 token id 不一致的问题。

    部分模型（如 llama 系列）的 config 里特殊 token id 可能缺失或与
    tokenizer 实际编码结果不一致，这里统一以 tokenizer 编码结果为准，
    同步写回 tokenizer 和 `model.config`。

    参数:
        model: 待同步特殊 token id 的模型，原地修改 `model.config`。
        tokenizer: 提供特殊 token 定义的分词器，原地修改其 token id 属性。

    返回:
        None，所有修改均原地生效。
    """
    for name in ['bos', 'eos', 'pad', 'unk']:
        
        token_id_name = '_'.join([name, 'token_id'])
        token_name = '_'.join([name, 'token'])

        token_id = getattr(tokenizer,  token_id_name)
        if token_id is None:
            token_str = getattr(tokenizer,  token_name)
            token_id = tokenizer.encode(token_str, add_special_tokens=False)[0]

        setattr(tokenizer, token_id_name, token_id)
        setattr(model.config, token_id_name, token_id)


def train():
    """有监督微调训练入口。

    解析命令行传入的模型/数据/训练参数，加载基础模型与 tokenizer，
    按需补齐特殊 token、启用 LoRA 或冻结部分参数，构建训练数据集后
    调用 `transformers.Trainer` 执行训练，并保存训练状态与最终模型。
    该函数不接受显式参数，所有配置均通过命令行参数传入；无返回值，
    产出以模型文件和日志的形式落盘。
    """
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print_rank(f"Loading model from {model_args.model_name_or_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        use_cache=False
    )
    if model_args.gradient_checkpointing_enable:
        print_rank("gradient_checkpointing_enable")
        model.gradient_checkpointing_enable()
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )

    print_rank("Model class:", model.__class__)
    print_rank("Tokenizer class:", tokenizer.__class__)

    # 只更改了bos_token和eos_token的文本显示，但是bos_token和eos_token自己会对应到1，2
    special_tokens_dict = dict()
    if not tokenizer.pad_token:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if not tokenizer.eos_token:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if not tokenizer.bos_token:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if not tokenizer.unk_token:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    if special_tokens_dict:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

        update_token_id(model, tokenizer)
    
    # print special token
    attrs =  ["pad_token", "pad_token_id", "bos_token", "bos_token_id", "eos_token", "eos_token_id", "unk_token", "unk_token_id", "model_max_length"]
    tokenizer_attrs, model_attrs = list(), list()
    for attr in attrs:
        if hasattr(tokenizer, attr):
            tokenizer_attrs.append(getattr(tokenizer, attr))
        else:
            tokenizer_attrs.append("-")
        if hasattr(model.config, attr):
            model_attrs.append(getattr(model.config, attr))
        else:
            model_attrs.append("-")

    values = [[""] + attrs, ["tokenizer"] + tokenizer_attrs, ["model"] + model_attrs]

    def plot_table(values):
        n_col = len(values[0])
        len_col = [max([len(str(row[i_col])) for row in values] + [6]) for i_col in range(n_col)]
        print_rank("+" + "+".join(["".center(len_v, "-") for len_v in len_col]) + "+")
        for row in values:
            print_rank("|" + "|".join([str(v).center(len_v, " ") for v, len_v in zip(row, len_col)]) + "|")
            print_rank("+" + "+".join(["".center(len_v, "-") for len_v in len_col]) + "+")
    
    plot_table(values)

    # enable lora according to the arguments
    if training_args.enable_lora:
        from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha, 
            target_modules = training_args.target_modules,
            lora_dropout=training_args.lora_dropout
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    assert not (training_args.only_update_ffn and training_args.only_update_attn), "Cannot open only_update_ffn and only_update_attn at the same time!"
    
    if training_args.only_update_ffn:
        # Turn off requires_grad for parameters with prefix "mlp"
        for name, p in model.named_parameters():
            if "mlp" in name:
                p.requires_grad = False
    
    if training_args.only_update_attn:
        # Turn off requires_grad for parameters with "self_attn" in its name
        for name, p in model.named_parameters():
            if "self_attn" in name:
                p.requires_grad = False

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    print_rank("Finish training...")


if __name__ == "__main__":
    train()
