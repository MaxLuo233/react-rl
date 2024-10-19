from react_env import ReActEnv
# from react_env import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
from collections import deque
import time
from tqdm import tqdm
import random
import pickle
import json
import deepspeed
import pdb

#CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed ppo_train_action_head.py

ds_config = {
    "train_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "fp16": {
        "enabled": True
    },
    "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00001,
      "betas": [
        0.8,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 3e-7
        }
    },
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True
    }
}

def load_data(react_file_path, cache_path, cluster_path):
    with open(react_file_path,'r') as f:
        lines = f.readlines()
    react_data = []
    for line in lines:
        line = json.loads(line)
        react_data.append(line)
    with open(cache_path,'rb') as f:
        dict_action_embedding = pickle.load(f)
    with open(cluster_path,'r') as f:
        dict_action_cluster_label = json.loads(f.read())
    return react_data, dict_action_embedding, dict_action_cluster_label


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

react_file_path = "/home/lzq/react-rl/action-level/action-head/react_prompt/new_react_prompt.jsonl"
cache_path = "embedding_cache.pkl"
cluster_path = "cluster_actions.json"
list_react_data, dict_action_embedding, dict_action_cluster_label = load_data(react_file_path, cache_path, cluster_path)
dict_label_embedding = {}
for action in list(dict_action_embedding.keys()):
    if dict_action_cluster_label[action] not in dict_label_embedding:
        dict_label_embedding[dict_action_cluster_label[action]] = []
    else:
        dict_label_embedding[dict_action_cluster_label[action]].append(dict_action_embedding[action])
for label in list(dict_label_embedding.keys()):
    dict_label_embedding[label] = np.mean(np.array(dict_label_embedding[label]),axis=0)
print("-----Load Data Done-----")


from mistral_with_ppo_head import MistralPPOHeadModel
from transformers import LlamaTokenizerFast, AutoTokenizer, AutoModelForCausalLM, MistralConfig
from ppo_head import PPOHead

save_dir = "/data2/lzq/mistral-7B-action-level-PPO"
import os
if os.path.exists(save_dir):
    os.system(f"rm -rf {save_dir}")   
os.makedirs(save_dir, exist_ok=True)

model_path = "/share/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24/"
config = MistralConfig.from_pretrained(model_path)

agent_model = MistralPPOHeadModel(config=config)#, MistralPPOHeadModel(config=config)
agent_model_embedding_layer = agent_model.model.embed_tokens
embedding_dim = agent_model.model.embed_tokens.embedding_dim
# value_model.set_action_head(PPOHead(input_dim=embedding_dim, output_dim=1))

agent_tokenizer = AutoTokenizer.from_pretrained(model_path)
agent_model = agent_model.from_pretrained(model_path, torch_dtype=torch.float16)

if agent_tokenizer.pad_token is None:
    agent_tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    agent_model.resize_token_embeddings(len(agent_tokenizer))
print("-----Load Model Done-----")


# num_episodes = 10
num_steps = 8 # 20; 8 for toy test
learning_rate = 1e-5
# total_timesteps = 40
num_envs = 2 # 256; 2 for toy test
batch_size = num_steps * num_envs # 8*2 for toy test
minibatch_size = 2 # 64; 2 for toy test
anneal_lr = True

gae = True
gamma = 0.99
gae_lambda = 0.95

update_epochs = 3

clip_coef = 0.2
norm_adv = True
clip_vloss = True
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
target_kl = None
# Initialize
obs = []
for i in range(num_steps):
    obs.append(['']*num_envs)

actions = torch.zeros((num_steps, num_envs)).to(device)
logprobs = torch.zeros((num_steps, num_envs)).to(device)
rewards = torch.zeros((num_steps, num_envs)).to(device)
dones = torch.zeros((num_steps, num_envs)).to(device)
values = torch.zeros((num_steps, num_envs)).to(device)
# avg_returns = deque(maxlen=20)

# start the game
global_step = 0
start_time = time.time()
num_updates = 20 # 40/8=5 for toy test

update = 1
# 构建环境列表
envs = []
embedding_initial_states = []
list_inputs = []
for react_data in list_react_data[(update-1)*num_envs:(update)*num_envs]:
    env = ReActEnv(react_data, dict_action_embedding, dict_action_cluster_label, dict_label_embedding)
    inputs = agent_tokenizer(env.state)
    envs.append(env)

next_obs = [''] * num_envs
for env in envs:
    index = envs.index(env)
    next_obs[index] = env.state
next_done = torch.zeros(num_envs).to(device)

agent_model.set_action_head(PPOHead(input_dim=embedding_dim, output_dim=100)) # 先加载后set_action_head
# agent_model.cuda()
# agent_model = nn.DataParallel(agent_model, device_ids =[0,1,2,3])
model_engine, optimizer, _, _ = deepspeed.initialize( #改为deepspeed
    model=agent_model,
    model_parameters=agent_model.parameters(),
    config=ds_config
)

# optimizer = optim.Adam(agent_model.parameters(), lr=learning_rate, eps=1e-5)
# Annealing the rate if instructed to do so.
if anneal_lr:
    frac = 1.0 - (update - 1.0) / num_updates
    lrnow = frac * learning_rate
    # optimizer.param_groups[0]["lr"] = lrnow

# for update in tqdm(range(num_updates)):
#     print(f"============={update} update==============")

for step in tqdm(range(0, num_steps)): 
    global_step += 1
    obs[step] = next_obs
    dones[step] = next_done
    # 执行action
    with torch.no_grad():
        print(next_obs)
        encoded_inputs = agent_tokenizer(next_obs, padding=True, truncation=True, max_length=1024, return_tensors='pt')
        encoded_inputs = encoded_inputs.to(device)
        outputs = agent_model(**encoded_inputs) 
        logits = outputs.logits.squeeze(1) # 输出维度[batch_env_size, 1, prediction_size]
        probs = Categorical(logits=logits)
        action = probs.sample() # 多个环境同时采样
        logprob = probs.log_prob(action)
        # action = torch.max(logits.squeeze(1), dim=-1)
        # value = value_model(**encoded_inputs)
        # values[step] = value.flatten()
    actions[step] = action
    logprobs[step] = logprob
    for i, env in enumerate(envs):
        next_obs_, reward, done_, _ = env.step(action[i].cpu().item()) # 环境交互
        rewards[step][i] = torch.tensor(reward).to(device).view(-1)
        next_obs[i], next_done[i] = next_obs_, torch.tensor(done_).to(device)

    with torch.no_grad(): # 不含value计算步骤，可省略
        if gae:
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                delta = rewards[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages 
        else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + gamma * nextnonterminal * next_return
            advantages = returns

b_obs = [item for sublist in obs for item in sublist]
b_logprobs = logprobs.reshape(-1,).to(device) # 相当于flatten为一维数组
b_actions = actions.reshape(-1,).to(device)
b_advantages = advantages.reshape(-1,).to(device)
b_returns = returns.reshape(-1,).to(device)

# PPO Optimizing the policy and value network
b_inds = np.arange(batch_size) # batch_size = num_steps * num_envs
clipfracs = []

model_engine.train()
for epoch in tqdm(range(update_epochs)):
    np.random.shuffle(b_inds)
    for start in range(0, batch_size, minibatch_size):
        end = start + minibatch_size
        mb_inds = b_inds[start:end]
        mb_obs = b_obs[start:end]
        encoded_inputs = agent_tokenizer(mb_obs, padding="max_length", truncation=True, max_length=1024, return_tensors='pt')
        encoded_inputs = encoded_inputs.to(device) #minibatch=2
        # outputs = agent_model(**encoded_inputs)
        # pdb.set_trace()
        outputs = model_engine(**encoded_inputs) 
        logits = outputs.logits.squeeze(1) # 输出维度[batch_env_size, 1, prediction_size]
        probs = Categorical(logits=logits)
        newlogprob = probs.log_prob(b_actions[mb_inds]) # 多个环境同时采样
        entropy = probs.entropy()
        # _, newlogprob, entropy, newvalue = agent_model.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
        logratio = newlogprob - b_logprobs[mb_inds]
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

        mb_advantages = b_advantages[mb_inds] # 截取minibatch
        if norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss

        # optimizer.zero_grad()
        # loss.backward()
        # nn.utils.clip_grad_norm_(agent_model.parameters(), max_grad_norm)
        # optimizer.step()
        model_engine.backward(loss)
        nn.utils.clip_grad_norm_(agent_model.parameters(), max_grad_norm)
        model_engine.step()

    if target_kl is not None:
        if approx_kl > target_kl:
            break

# 保存fp16的方法
# model_engine.save_16bit_model(save_dir, 'model.bin')


model_engine.save_checkpoint(save_dir)
# 需要在ckp路径下运行python3 zero_to_fp32.py . pytorch_model.bin

agent_tokenizer.save_pretrained(save_dir)
# torch.save(agent.state_dict(), "RLAgent.bin")