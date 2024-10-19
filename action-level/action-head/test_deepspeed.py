import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed
import pdb

# 加载预训练模型和Tokenizer
model_path = "/share/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24/"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    model.resize_token_embeddings(len(tokenizer))
# 加载示例数据集
dataset = load_dataset("./mrpc")

# 对输入进行tokenization
def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=1024)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"].remove_columns(["sentence1", "sentence2", "label", "idx"]).with_format("torch")

# import deepspeed

# 配置DeepSpeed
ds_config = {
    "bf16": {
        "enabled": True
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5,
            "betas": [0.98, 0.999],
            "eps": 1e-9
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 2e-5,
            "warmup_max_lr": 5e-5,
            "warmup_num_steps": 300
        }
    },
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e8,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        }
    },
    "gradient_accumulation_steps": 16,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False,
    "steps_per_print": 50
}



engine, _, _, _ = deepspeed.initialize(
        config=ds_config,
        model=model, 
        model_parameters=model.parameters(),
    )
engine.train()

# 数据加载器
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4)

# 训练循环
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(3):
    for batch in train_dataloader:
        # 将数据移动到GPU
        inputs = {k: v.to(device) for k, v in batch.items()}
        # pdb.set_trace()
        # 向前传播
        outputs = engine(**inputs)
        loss = outputs.loss

        # 反向传播
        engine.backward(loss)
        engine.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
