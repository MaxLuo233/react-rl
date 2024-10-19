import numpy as np
from FlagEmbedding import BGEM3FlagModel
import pickle

def load_cache(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def save_cache(file_path, cache):
    with open(file_path, 'wb') as f:
        pickle.dump(cache, f)


def load_bge_m3_model(model_name_or_path):
    bge_m3_model = BGEM3FlagModel(model_name_or_path, use_fp16=True)
    return bge_m3_model

# load model
Embedding_model_path = "/share/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181/"
Embedding_cache_path = "/home/lzq/OpsTroubleshootLLM/eval/embedding_cache.pkl"
Embedding_model = load_bge_m3_model(Embedding_model_path)
Embedding_cache = load_cache(Embedding_cache_path)

def encode_sentence(sentence, model=Embedding_model, embedding_cache=Embedding_cache):
    if embedding_cache and sentence in embedding_cache:
        return embedding_cache[sentence]
    return model.encode(sentence)['dense_vecs'] # dim_size=1024

def get_similarity(s1, s2):
    embedding1, embedding2 = encode_sentence(s1), encode_sentence(s2)
    return embedding1 @ embedding2.T

def find_choice_with_max_similarity(gt, choices):
    list_embedding = [get_similarity(c, gt) for c in choices]
    if gt in choices:
        return choices.index(gt), gt, list_embedding
    index = list_embedding.index(max(list_embedding))
    return index, choices[index], list_embedding

def retrieval_topK_choices(prompt, skip_actions, dict_candidate_embeddings=Embedding_cache, model=Embedding_model, topK=4, ):
    prompt_embedding = encode_sentence(prompt, model)

    similarities = {}
    for action, candidate_embedding in dict_candidate_embeddings.items():
        if action in skip_actions:
            continue
        similarities.update({action: prompt_embedding @ candidate_embedding.T})

    sorted_embeddings = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    
    if topK == 1:
        return sorted_embeddings[0][0]
    else:
        return [c[0] for c in sorted_embeddings[:topK]]

Helper_model_path = "/data2/lzq/ft_multi_choice_0820_Qwen2_7B_Instrcut_huawei_epoch_1"
from transformers import AutoModelForCausalLM, AutoTokenizer
Helper_tokenizer = AutoTokenizer.from_pretrained(Helper_model_path)
Helper_model = AutoModelForCausalLM.from_pretrained(Helper_model_path).to("cuda")

import torch
def inference(prompt, choices, model=Helper_model, tokenizer=Helper_tokenizer):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    logits = model(input_ids).logits[:, -1].flatten()

    list_default_choices = ["A", "B", "C", "D"]
    list_choices = choices
    probs = (
        torch.nn.functional.softmax(
            torch.tensor(
                [logits[tokenizer(choice, add_special_tokens=False).input_ids[-1]] for choice in list_choices]
            ),
            dim=0,
        ).detach().cpu().to(torch.float32).numpy())
    answer = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
    prediction_dict = {_[0]: _[1] for _ in zip(list_default_choices, probs.tolist())}
    return answer, prediction_dict

def generate_prompt(question, choices):
    prefix = """你是一个故障检测智能助手。在下列故障定位问题中，需要从候选选项中选择最佳行动，来进行故障定位。

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
            Choices:
            A. xxx
            B. xxx  
            C. xxx
            D. xxx
            Answer: B
            """
    prompt = (prefix + 'Question 2:\n  ' 
              + '\n'.join(question.split('\n')[1:]) + '\nChoices:\n' 
              + '  A. ' + choices[0] + '\n'
              + '  B. ' + choices[1] + '\n'
              + '  C. ' + choices[2] + '\n'
              + '  D. ' + choices[3])
    return prompt


class ReActEnv():
    def __init__(self, question:str, choices:list, gt_action:str):
        self.question = question

        self.raw_state =[question, choices, ""]     # question, choices, last chosen:str|None
        self.action_space = ['subst_1', 'subst_all', 'choose']
        self.action = None
        self.reward = None
        self.done = 0 # 不用布尔值，便于转化为张量
        self.current_step = 0
        self.max_step = 500
        
        self.gt = gt_action
        self.skip_choices = []
        self.embedding_state = encode_sentence(generate_prompt(self.raw_state[0], self.raw_state[1]))
    
    def reset(self, **data):
        question = data.get('question', None)
        choices = data.get('choices', [])
        self.raw_state = [question, choices, False]
        self.action = None
        self.reward = None
        self.current_step = 0
        return

    def step(self, action):
        assert action in [0, 1, 2]

        if self.done:
            self.current_step += 1
            return self.embedding_state, self.reward, self.done, self.current_step
        if self.current_step > self.max_step:
            raise ValueError('Steps Exceed')
        
        if action == 0:
            self.subst_1()
        elif action == 1:
            self.subst_all()
        else:
            self.choose()

        self.embedding_state = encode_sentence(generate_prompt(self.raw_state[0], self.raw_state[1]))
        self.current_step += 1
        return self.embedding_state, self.reward, self.done, self.current_step
    
    def close(self,):
        pass

    def get_reward(self,): # 根据state计算reward，如果根据action则计算分散到各action方法中
        choices, chosen = self.raw_state[1], self.raw_state[2]
        if chosen and chosen == self.gt:
                self.reward = 1
                self.done = 1
        elif chosen and chosen!=self.gt and self.gt in choices: # 选错
                self.reward = -1
        else:
            index, choice, list_similarities = find_choice_with_max_similarity(self.gt, choices)
            self.reward = -1 + sum(list_similarities)/4
        return self.reward


    def subst_1(self,): # 存疑，可能在chosen正确答案后替换掉已有的正确答案
        choices, target_choice = self.raw_state[1], self.raw_state[2]
        if not target_choice:
            target_choice = find_choice_with_max_similarity(self.gt, choices)[1] 
        self.skip_choices.append(target_choice)
        new_choice = retrieval_topK_choices(
                        '\n'.join(self.question.split('\n')[1:]), 
                        skip_actions=self.skip_choices, 
                        topK=1)
        choices.remove(target_choice)   
        choices.append(new_choice)
        self.raw_state[1] = choices
        self.raw_state[2] = None
        self.reward = self.get_reward()
        return

    def subst_all(self,):
        old_choices = self.raw_state[1]
        self.skip_choices += old_choices
        new_choices = retrieval_topK_choices(
                        '\n'.join(self.question.split('\n')[1:]), 
                        skip_actions=self.skip_choices, 
                        topK=4)
        self.raw_state[1] = new_choices    
        self.raw_state[2] = None
        self.reward = self.get_reward()
        return

    def choose(self,):
        question = self.question
        choices = self.raw_state[1]
        prompt = generate_prompt(question, choices)
        ans = inference(prompt, choices)[0]
        self.raw_state[2] = choices[['A','B','C','D'].index(ans)]
        self.reward = self.get_reward()       
        return
    

if __name__ == "__main__":
    test_env = ReActEnv('\n你是一个故障检测智能助手。在下列故障定位问题中，需要从候选选项中选择最佳行动，来进行故障定位。\nAlert:100002 微服务 Running 状态的容器数小于 1\nAction:',
                        ['检查微服务配置', '考虑设置任务直接失败而不挂起以提高故障发现效率', '再次检查微服务的运行状态和容器数目', '利用告警中的定位信息，登录至对应的master节点，检查服务状态'],
                        '检查微服务配置')