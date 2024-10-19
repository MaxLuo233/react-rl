import gym
from gym import Env
import math, random
import os
import json
import numpy as np
from FlagEmbedding import BGEM3FlagModel

bge_m3_model_name_or_path = "/share/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
bge_m3_model = BGEM3FlagModel(bge_m3_model_name_or_path, use_fp16=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model = AutoModelForCausalLM.from_pretrained("/data2/lzq/finetune-qwen2-7B-instruct")
tokenizer = AutoTokenizer.from_pretrained("/data2/lzq/finetune-qwen2-7B-instruct")
model.cuda()

inference_content ="""
你是一个故障检测智能助手。在下列故障定位问题中，需要从候选选项中选择最佳行动，来进行故障定位。

每个问题中，包含了故障告警和已进行的故障定位操作，格式如下：

Alert:你要处理的告警信息
Action:你应该采取的行动
Action Input:行动的需要的具体参数，以JSON格式的Key-Value字典表示
Thought:你应当解释采取所选行动的原因
Observation:行动中观察到的结果
... (上述Action/Action/Action Input/Thought/Observation可以重复多次，但Alert不会重复)

问题列表：
Question 1:
  Alert:虚拟机公网EIP不通
  Action: xxxx
  Action Input:{"cmd": "ifconfig -a"}
  Thought:虚拟机要访问公网，应当具有IP地址，所以应当先检查是否正确配置了IP地址
  Observation:有IP地址
Answer:查找公网IP
"""

def encode_sentence(sentence, embedding_cache=None, model=bge_m3_model):
    if embedding_cache and sentence in embedding_cache:
        return embedding_cache[sentence]
    return model.encode(sentence)['dense_vecs']

def retrieval_simlarities_and_action(model_answer, choices):
    model_answer_embedding = encode_sentence(model_answer)
    choices_embedding = {choice: encode_sentence(choices[choice]['content']) for choice in list(choices.keys())}
    similarities = {choice: choices_embedding[choice] @ model_answer_embedding.T for choice in list(choices_embedding.keys())}
    model_choice = sorted(similarities.items(), key=lambda p: p[1], reverse=True)[0][0]
    return similarities, model_choice

class Activate_Env():

    def __init__(self):
        # self.env = gym.make('myenv')
        # self.env.configure_from_data(multi_choice_data)
        self.current_step = 1
        self.action = None
        self.choices = {}
        self.similarities = {}
        self.gt_action = None
        self.rewards = 0
        self.base_true_reward = None #选对的奖励
        self.false_records = []
        self.done = False
        
        self.discount = 0.7
        self.max_steps = 10

    def model_generate(self, multi_choice_data):

        lines = ['  ' + line.strip() for line in multi_choice_data['retrieval_prompt'].strip('\n').split('\n')]
        question_content = "\nQuestion 2:\n" + '\n'.join(lines[1:])

        if not len(self.false_records):
            inference_content_prompt = inference_content + question_content
        else:
            for id, r in enumerate(self.false_records):
                question_content = question_content + r + f"\n第{id}次回答\'{r}\'错误，请重新回答" + question_content #需要修改
            inference_content_prompt = inference_content + question_content

        input_ids = tokenizer(inference_content_prompt, return_tensors='pt').input_ids.cuda()
        generated_token_ids = model.generate(input_ids, max_new_tokens=50) #维度为batch_size*sequence_length
        model_answer = tokenizer.decode(generated_token_ids[0], skip_special_tokens=True)
        
        return inference_content_prompt, model_answer


    def configure_from_data(self, multi_choice_data, model_answer): #确定self.action, self.gt_action, self.choices, self.similarities, self.base_reward
        choices = multi_choice_data['choices']
        self.choices = {c: choices[c]['content'] for c in list(choices.keys())} 
        self.similarities, self.action = retrieval_simlarities_and_action(model_answer, choices)
        if self.false_records is not None: #随机选择
            self.action = random.sample(['A','B','C','D'], 1)
        self.gt_action = multi_choice_data["gt_action_choice"]

        true_similarity = math.log(self.similarities[self.gt_action])
        false_similarities = [math.log(self.similarities[c]) for c in list(self.similarities.keys()) if c != self.gt_action]
        self.base_true_reward = true_similarity - sum(false_similarities)


    def step(self):
        if self.done or self.current_step > self.max_steps:
            raise ValueError('Task is already done!')
        
        if self.action == self.gt_action:
            self.rewards += self.base_true_reward * pow(self.discount, self.current_step-1)
            self.current_step += 1
            self.done = True
        else:  
            self.rewards -= (math.log(self.similarities[self.action]) - math.log(self.similarities[self.gt_action])) * pow(1/self.discount, self.current_step-1)
            self.false_records.append(self.choices[self.action])
            self.current_step += 1
        
        return self.action, self.choices[self.action], self.rewards, self.current_step, self.done


    def generate_trajectory(self, multi_choice_data):
        self.reset()
        records = []

        while not self.done and self.current_step <= self.max_steps:
            prompt, model_answer = self.model_generate(multi_choice_data)
            self.configure_from_data(multi_choice_data, model_answer)
            self.action, self.choices[self.action], self.rewards, self.current_step, self.done = self.step() #第一次回答错误后self.action需要修改为随机选取
            records.append({'prompt':prompt, 'answer':model_answer, 'chosen_action': self.choices[self.action], 'similarity': self.similarities[self.action],
                                 'gt_action': self.choices[self.gt_action], 'reward':self.rewards, 'step':self.current_step})
            print(self.similarities, self.choices[self.action], self.choices[self.gt_action], self.rewards)
        return records

    def reset(self):
        self.current_step = 1
        self.action = None
        self.choices = {}
        self.similarities = {}
        self.gt_action = None
        self.rewards = 0
        self.base_true_reward = None #选对的奖励
        self.false_records = []
        self.done = False

    def main(self, input_dir="/home/lzq/OpsTroubleshootLLM/output/multi_choices_react_question/react_questions.jsonl", 
             output_dir="/home/lzq/react-rl/action-level/react_data_with_rewards.jsonl"):
        with open(input_dir, 'r') as fin, open(output_dir, 'a') as fout:
            lines = fin.readlines()
            for line in lines:
                def custom_serializer(obj):
                    if isinstance(obj, np.float16):
                        return float(obj)
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
                
                multi_choice_data = json.loads(line)
                records = self.generate_trajectory(multi_choice_data)
                for r in records:
                    fout.write(json.dumps(r, default=custom_serializer))
                    fout.write('\n')


if __name__ == '__main__':
    env = Activate_Env()
    env.main()