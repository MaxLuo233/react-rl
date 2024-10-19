import numpy as np
# from FlagEmbedding import BGEM3FlagModel
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


# load model
Embedding_model_path = "/share/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181/"
Embedding_cache_path = "/home/lzq/OpsTroubleshootLLM/eval/embedding_cache.pkl"
# Embedding_model = BGEM3FlagModel.from_pretrained(Embedding_model_path)
Embedding_model = None 
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
# Helper_model = AutoModelForCausalLM.from_pretrained(Helper_model_path).to("cuda")
Helper_model = None

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


class ReActEnv():
    def __init__(self, react_data:dict, 
                 dict_action_embedding:dict, 
                 dict_action_cluster_label:dict,
                 dict_label_embedding:dict):
        self.alarm = react_data['alarm_data'][0]
        self.round_prompts = react_data['round_prompts']
        self.round_actions = react_data['round_actions']
        self.action_embedding = dict_action_embedding
        self.action_cluster_label = dict_action_cluster_label
        self.label_embedding = dict_label_embedding
        self.round = 0

        self.state = react_data['round_prompts'][0]
        self.action_space = range(100)
        self.action = None
        self.reward = None
        self.done = 0 # 不用布尔值，便于转化为张量
        self.current_step = 0
        self.max_step = 500
        
        self.skip_choices = []
        # self.embedding_state = encode_sentence(generate_prompt(self.raw_state[0], self.raw_state[1]))
    
    def reset(self, **react_data):
        self.alarm = react_data['alarm_data'][0]
        self.round_prompts = react_data['round_prompts']
        self.round_actions = react_data['round_actions']
        self.state = self.alarm
        self.action = None
        self.reward = None
        self.current_step = 0
        return

    def step(self, action_label):
        assert action_label in range(100)
        if self.done:
            self.current_step += 1
            return self.state, self.reward, self.done, self.current_step
        if self.current_step > self.max_step:
            raise ValueError('Steps Exceed')
        
        gt_action = self.round_actions[self.round]
        gt_action_label = self.action_cluster_label[gt_action]
        if action_label == gt_action_label:
            self.reward = 1
            self.round += 1
            self.state = self.round_prompts[self.round]
        else:
            action_label_embedding = self.label_embedding[action_label]
            gt_action_embedding = self.action_embedding[gt_action]
            similarity = -1 + action_label_embedding @ gt_action_embedding.T
            self.reward = similarity

        self.current_step += 1
        return self.state, self.reward, self.done, self.current_step
    
    def close(self,):
        pass
    

if __name__ == "__main__":
    test_env = ReActEnv('\n你是一个故障检测智能助手。在下列故障定位问题中，需要从候选选项中选择最佳行动，来进行故障定位。\nAlert:100002 微服务 Running 状态的容器数小于 1\nAction:',
                        ['检查微服务配置', '考虑设置任务直接失败而不挂起以提高故障发现效率', '再次检查微服务的运行状态和容器数目', '利用告警中的定位信息，登录至对应的master节点，检查服务状态'],
                        '检查微服务配置')