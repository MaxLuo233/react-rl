import sklearn
from sklearn.cluster import KMeans
from FlagEmbedding import BGEM3FlagModel
import pickle
import numpy as np
import hashlib
import re
import json

Embedding_model_path = "/share/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181/"
Embedding_model = BGEM3FlagModel(Embedding_model_path, use_fp16=True)
Embedding_cache_path = "/home/lzq/OpsTroubleshootLLM/eval/embedding_cache.pkl"

with open(Embedding_cache_path, 'rb') as f:
    Embedding_cache = pickle.load(f)

def encode_sentence(sentence, model=Embedding_model, embedding_cache=Embedding_cache):
    if embedding_cache and sentence in embedding_cache:
        return embedding_cache[sentence]
    return model.encode(sentence)['dense_vecs'] # dim_size=1024

def sha1(text: str):
    return hashlib.sha1(text.encode()).hexdigest()

def parse_alarm_to_react(alarm_data, i):
    thought = re.search(r'(?<=Thought:).*?(?=;\s*Action:|；\s*Action:)', alarm_data[i]).group()
    action = re.search(r'(?<=Action:).*?(?=;\s*Observation:|；\s*Observation:)', alarm_data[i]).group()
    observation = re.search(r'(?<=Observation:).*', alarm_data[i]).group()
    return thought, action, observation

def get_react_data_action(react_data_file):
    list_step_json = []
    list_action_json = []
    with open(react_data_file,'r') as f:
        lines = f.readlines()
    for line in lines:
        line = eval(line)
        if len(line) != 2:
            continue
        skip = False
        round_steps = []
        round_actions = []
        alarm_name, alarm_data = line[0], line[1]
        alarm_sha1 = sha1(''.join(alarm_data))
        for i in range(len(alarm_data)):
            if alarm_data[i].find("Observation:") > 0:
                try:
                    thought, action, observation = parse_alarm_to_react(alarm_data, i)
                    round_steps.append({'thought':thought, 'action':action, 'observation':observation})
                    round_actions.append(action)
                except Exception as ex:
                    skip = True
                    break
            elif alarm_data[i].find("Final Answer:") > 0:
                pass
            else:
                skip = True
                break
        step_json = {"alarm_sha1": alarm_sha1, "alarm_name": alarm_name, "round_steps": round_steps, "skip": skip}
        action_json = {"alarm_sha1": alarm_sha1, "alarm_name": alarm_name, "round_actions": round_actions, "skip": skip}
        list_step_json.append(step_json)
        list_action_json.append(action_json)
    return list_step_json, list_action_json

def get_embedding_file(output_path='embedding_path'):
    list_step_json, list_action_json = get_react_data_action('processed_augment_react_data_list')
    print(len(list_action_json))

    dict_embeddings = {}
    for id,action_json in enumerate(list_action_json):
        round_actions = action_json["round_actions"]
        if id % 100 == 0:
            print(id)
        for action in round_actions:
            dict_embeddings.update({action:encode_sentence(action)}) 

    with open(output_path, 'wb') as f:
        pickle.dump(dict_embeddings, f)
        

if __name__ == '__main__':

    # with open('react_prompt/react_prompt.jsonl','r') as f:
    #     lines = f.readlines()
    # new_lines = []
    # for line in lines:
    #     line = json.loads(line)
    #     round_prompts = line['round_prompts']
    #     new_round_prompts = []
    #     for prompt in round_prompts:
    #         id1 = prompt.find('Action: xxxx')
    #         id2 = prompt.find('Choices:')
    #         id3 = prompt.find('Question 2:')
    #         new_prompt = (prompt[:id1] + 
    #                       'Action:检查虚拟机是否配置IP地址' +
    #                       prompt[id1+len('Action: xxxx'):id2] + '\n' +
    #                       prompt[id3:])
    #         new_round_prompts.append(new_prompt)
    #     line.update({'round_prompts': new_round_prompts})
    #     new_lines.append(line)
    
    # with open('react_prompt/new_react_prompt.jsonl','w') as fout:
    #     for line in new_lines:
    #         fout.write(json.dumps(line,ensure_ascii=False))
    #         fout.write('\n')
           
    with open('react_prompt/react_action.jsonl','r') as f:
        lines = f.readlines()
    list_actions = []
    for line in lines:
        line = json.loads(line)
        list_actions += line['round_actions']
    
    unique_list_actions = list(set(list_actions))

    dict_embeddings = {}
    with open('embedding_cache.pkl','wb') as f:
        for id, action in enumerate(unique_list_actions):
            if id % 100 == 0:
                print(id)
            dict_embeddings.update({action:encode_sentence(action)})
        pickle.dump(dict_embeddings,f)
    # with open('embedding_cache.pkl','rb') as f:
    #     dict_embeddings = pickle.load(f)

    list_actions = list(dict_embeddings.keys())
    vectors = np.array(list(dict_embeddings.values()))
    kmeans = KMeans(n_clusters=100, random_state=0).fit(vectors)
    labels = kmeans.labels_
    cluster_actions = {action: int(label) for action, label in zip(list_actions, labels)}
    import json
    with open('cluster_actions.json', 'w') as file:
        json.dump(cluster_actions, file, ensure_ascii=False)