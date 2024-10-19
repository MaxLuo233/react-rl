"""
python examples/scripts/reward_modeling.py \
    --model_name_or_path=facebook/opt-350m \
    --output_dir="reward_modeling_anthropic_hh" \
    --per_device_train_batch_size=16 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --eval_strategy="steps" \
    --eval_steps=500 \
    --max_length=512 \
"""
import warnings
import sys
import os

import torch
from accelerate import PartialState
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, 
                          AutoModelForCausalLM,
                          AutoTokenizer, HfArgumentParser, TrainingArguments)
from dataclasses import dataclass, field
from typing import *

from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config


tqdm.pandas()
torch.cuda.empty_cache()


@dataclass
class RewardConfig(TrainingArguments):
    """
    RewardConfig collects all training arguments related to the [`RewardTrainer`] class.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int`, *optional*, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        gradient_checkpointing (`bool`, *optional*, defaults to `True`):
                If True, use gradient checkpointing to save memory at the expense of slower backward pass.
    """

    max_length: Optional[int] = None
    """The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator."""
    dataset_num_proc: Optional[int] = None
    """Coefficient to incentivize the reward model to output mean-zero rewards (proposed by https://huggingface.co/papers/2312.09244, Eq. 2). Recommended value: `0.01`."""
    center_rewards_coefficient: Optional[float] = None

    prompt_prefix: Optional[str] = field(default=None, metadata={"help": "the system prompt added to dataset"})

    data_dir : Optional[str] = field(default="./data/ppo_dpo_react_data_datasetdict", metadata={"help": "the training and validation dataset directory"})


@dataclass
class ModelConfig:
    """
    Arguments which define the model and tokenizer to load.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The model checkpoint for weights initialization.")},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code when loading a model."})
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which attention implementation to use; you can run --attn_implementation=flash_attention_2, in which case you must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )
    lora_task_type: str = field(
        default="CAUSAL_LM", metadata={"help": "The task_type to pass for LoRA (use SEQ_CLS for reward modeling)"}
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": (
                "Use Rank-Stabilized LoRA (https://huggingface.co/papers/2312.03732), which sets the adapter "
                "scaling factor to lora_alpha/√r, instead of the original default value of `lora_alpha/r`."
            )
        },
    )
    load_in_8bit: bool = field(
        default=False, metadata={"help": "use 8 bit precision for the base model - works only with LoRA"}
    )
    load_in_4bit: bool = field(
        default=False, metadata={"help": "use 4 bit precision for the base model - works only with LoRA"}
    )

    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")

        if isinstance(self.lora_target_modules, list) and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]

if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        config, model_config = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        config, model_config = parser.parse_args_into_dataclasses()

    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch_dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True, 
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code, ignore_mismatched_sizes=True, **model_kwargs
    )
    model.config.pad_token_id = model.config.eos_token_id

    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    ################
    # Dataset
    ################
    data_dir = config.data_dir
    prompt_prefix = config.prompt_prefix
    train_dataset = load_from_disk(os.path.join(data_dir, 'train'))
    validation_dataset = load_from_disk(os.path.join(data_dir, 'validation'))
    # Tokenize chosen/rejected pairs of inputs
    # Adapt this section to your needs for custom datasets

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "prompt": [prompt_prefix + question + "\n\n" for question in samples["failure"]],
            "chosen": samples["pos"],
            "rejected": samples["neg"],
        }


    def tokenize_function(examples):
        
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen, max_length=512, padding="max_length", truncation=True, return_tensors='pt') 
            tokenized_rejected = tokenizer(rejected, max_length=512, padding="max_length", truncation=True, return_tensors='pt')

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"].squeeze(0))
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"].squeeze(0))
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"].squeeze(0))
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"].squeeze(0))

        return new_examples

    
    def preprocess_function(raw_datasets):
        raw_datasets = raw_datasets.map(
                    return_prompt_and_responses,
                    batched=True,
                    num_proc=config.dataset_num_proc,
                    remove_columns=raw_datasets.column_names,
                )
        
        raw_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=config.dataset_num_proc,
        )
 
        raw_datasets = raw_datasets.filter(
            lambda x: len(x["input_ids_chosen"]) <= config.max_length
            and len(x["input_ids_rejected"]) <= config.max_length,
            num_proc=config.dataset_num_proc,
        )
        return raw_datasets

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = preprocess_function(train_dataset)
        validation_dataset = preprocess_function(validation_dataset)

    ################
    # Training
    ################
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    # trainer.push_to_hub()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)