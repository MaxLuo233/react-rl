# import pudb; pu.db
import shutil
import os
import sys
from typing import *

from accelerate import PartialState
from datasets import load_dataset, load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    Qwen2ForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
import torch

from trl import ModelConfig
from trl.trainer.ppov2_trainer import PPOv2Config, PPOv2Trainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler


"""
python -i examples/scripts/ppo/ppo.py \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --non_eos_penalty \

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/ppo/ppo.py \
    --output_dir models/minimal/ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --local_rollout_forward_batch_size 1 \
    --deepspeed3 \
    --non_eos_penalty \

CUDA_VISIBLE_DEVICES="1,2,3" accelerate launch --num_processes=3 ppo_train.py ppo_train_cfg.json
"""


if __name__ == "__main__":
    parser = TrlParser((PPOv2Config, ModelConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        config, model_config = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        config, model_config = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(config.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE

    model_kwargs = dict(
        revision=model_config.model_revision,
        # device_map="auto",
        torch_dtype=torch.float16
    )

    # Accelerator会自动在GPU上加载模型
    value_model = Qwen2ForSequenceClassification.from_pretrained(
        config.reward_model_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs, num_labels=1
    )
    reward_model = Qwen2ForSequenceClassification.from_pretrained(
        config.reward_model_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs, num_labels=1
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs,
    )
    policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs,
    )
    ################
    # Dataset
    ################
    def get_react_data(data_dir:str, prompt_prefix:str, 
                       sanity_check:bool=False, cache_dir:Optional[str]=None, num_proc:int=4) -> Dataset:
        """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

        The dataset is converted to a dictionary with the following structure:
        {
            'prompt': List[str],
            'chosen': List[str],
            'rejected': List[str],
        }

        Prompts are structured as follows:
        "Question: " + <prompt> + "\n\nAnswer: "
        """
        dataset = load_from_disk(data_dir)
        
        original_columns = dataset.column_names

        if sanity_check:
            dataset = dataset.select(range(min(len(dataset), 1000)))

        def return_prompt_and_responses(samples) -> Dict[str, str]:
            return {
                "prompt": [prompt_prefix + question + "\n\n" for question in samples["failure"]],
                "chosen": samples["pos"],
                "rejected": samples["neg"],
            }
        
        return dataset.map(
            return_prompt_and_responses,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
            )
    
    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            content = element['prompt']+element['chosen'] if 'chosen' in element else element['prompt']+element['reject']
            outputs = tokenizer(
                element['prompt'],
                padding=True,
                return_tensors='pt'
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=1,
        )

    data_dir = config.data_dir
    prompt_prefix = config.prompt_prefix
    train_data_dir = os.path.join(data_dir, 'train')
    validation_data_dir = os.path.join(data_dir, 'validation')
    train_dataset = get_react_data(data_dir=train_data_dir, prompt_prefix=prompt_prefix, sanity_check=config.sanity_check)
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= config.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= config.max_length
    )

    # 3. Load evaluation dataset
    eval_dataset = get_react_data(data_dir=validation_data_dir, prompt_prefix=prompt_prefix, sanity_check=config.sanity_check)
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= config.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= config.max_length
    )

    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)


    ################
    # Training
    ################
    trainer = PPOv2Trainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    # trainer.generate_completions()