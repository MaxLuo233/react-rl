# pseudocode
# env = Env()
# agent = Agent()
# for episode in range(1, num_episodes):
#     next_obs = env.reset()
#     data = []
#     for step in range(1, max_episode_horizon):
#         obs = next_obs
#         action, other_stuff = agent.get_action(obs)
#         next_obs, reward, done, info = env.step(action)
#         data.append([obs, action, reward, done, other_stuff]) # store data
#         if done:
#             break
#     agent.learn(data)

# DRL Jalibreaking论文方法复现

from myenv import ReActEnv
from myagent import RLAgent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time
from tqdm import tqdm
import random
import pickle
import json

def load_data(mcq_file_path, cache_path):
    with open(mcq_file_path,'r') as f:
        lines = f.readlines()
    
    with open(cache_path, 'rb') as f:
        text = pickle.load(f)
    actions = list(text.keys())
    
    list_questions_and_choices = []
    for line in lines:
        line = json.loads(line)
        question = line['retrieval_prompt']
        choices = [c['content'] for c in list(line['choices'].values())]
        random_actions = random.sample(actions, 4)
        gt_choice = line['gt_action']
        list_questions_and_choices.append((question, choices, random_actions, gt_choice))
    return list_questions_and_choices


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mcq_file_path = "/home/lzq/OpsTroubleshootLLM/output/multi_choices_react_question/react_questions_4.jsonl"
cache_path = "/home/lzq/OpsTroubleshootLLM/eval/embedding_cache.pkl"
list_questions_and_choices = load_data(mcq_file_path, cache_path)
print("Load Data Done")

env = ReActEnv('\n你是一个故障检测智能助手。在下列故障定位问题中，需要从候选选项中选择最佳行动，来进行故障定位。\nAlert:100002 微服务 Running 状态的容器数小于 1\nAction:',
               ['检查微服务配置', '考虑设置任务直接失败而不挂起以提高故障发现效率', '再次检查微服务的运行状态和容器数目', '利用告警中的定位信息，登录至对应的master节点，检查服务状态'],
               '检查微服务配置')
                # 实际训练集多样性有待考虑
agent = RLAgent().to(device)

# num_episodes = 10
num_steps = 20 # 8 for toy test
learning_rate = 1e-5
# total_timesteps = 40
num_envs = 256 # 2 for toy test
batch_size = num_steps * num_envs # 8*2 for toy test
minibatch_size = 64 # 8 for toy test
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

optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

# Initialize
obs = torch.zeros((num_steps, num_envs, 1024)).to(device) # obs是state，需要先encode，最好在env完成
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

for update in (range(1, num_updates + 1)):
    # 构建环境列表
    envs = []
    for item in list_questions_and_choices[(update-1)*num_envs:(update)*num_envs]:
        r = random.choice([0,1])
        choices = item[1] if r else item[2]
        question, gt_action = item[0], item[3]
        envs.append(ReActEnv(question, choices, gt_action))

    next_obs = torch.zeros(num_envs, 1024).to(device)
    for env in envs:
        index = envs.index(env)
        next_obs[index] = torch.tensor(env.embedding_state, dtype=torch.float32).to(device)
    next_done = torch.zeros(num_envs).to(device)
    print(f"Update {update} Envs Initialize Done")

    # Annealing the rate if instructed to do so.
    if anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    # 采样数据
    for step in tqdm(range(0, num_steps)): 
        global_step += 1
        obs[step] = next_obs
        dones[step] = next_done
        # 执行action
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob
        for i, env in enumerate(envs):
            next_obs_, reward, done_, _ = env.step(action[i].cpu().numpy())
            rewards[step][i] = torch.tensor(reward).to(device).view(-1)
            next_obs[i], next_done[i] = torch.tensor(next_obs_, dtype=torch.float32).to(device), torch.tensor(done_).to(device)

            # if done:
            #     print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
            #     avg_returns.append(info["r"][idx])
            #     writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
            #     writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
            #     writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
            #     if np.average(avg_returns) > 17:
            #         writer.add_scalar("charts/time", time.time() - start_time, global_step)
            #         quit()

    with torch.no_grad(): # value计算步骤，可省略
        next_value = agent.get_value(next_obs).reshape(1, -1)
        if gae:
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + gamma * nextnonterminal * next_return
            advantages = returns - values

    # flatten the batch   envs>1时需要
    b_obs = obs.reshape(-1, 1024)
    b_logprobs = logprobs.reshape(-1,)
    b_actions = actions.reshape(-1,)
    b_advantages = advantages.reshape(-1,)
    b_returns = returns.reshape(-1,)
    b_values = values.reshape(-1,)
    
    # PPO Optimizing the policy and value network
    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in tqdm(range(update_epochs)):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
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

            # Value loss
            newvalue = newvalue.view(-1)
            if clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        if target_kl is not None:
            if approx_kl > target_kl:
                break

torch.save(agent.state_dict(), "RLAgent.pth")