import transformers
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(default="")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    lora_path: str | None = field(default="")


def convert_to_hf() -> None:
    """合并 LoRA 权重并导出 Hugging Face 格式模型。

    参数：无。模型和 LoRA 权重路径由命令行参数提供。
    返回值：无。合并后的模型和分词器保存到 ``output_dir``。
    """
    parser = transformers.HfArgumentParser(
            (ModelArguments, TrainingArguments)
        )
    model_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        training_args.lora_path,
        trust_remote_code=True,
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, training_args.lora_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    convert_to_hf()
