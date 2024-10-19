from datasets import Dataset, DatasetDict, load_from_disk
import random

with open('data/augment_react_data_huawei_positive_negative','r') as f:
    lines = f.readlines()

import pandas as pd

def dpo_data_generation():
    id = 0
    failure = ''
    data = {
        'id': [],
        'failure': [],
        'pos': [],
        'neg':[]
    }
    for line in lines:
        pos, neg = line.split('||')[0], line.split('||')[1]
        pos, neg = eval(pos), eval(neg)
        if pos[0] != failure:
            id += 1
        failure = pos[0]
        pos = ' '.join(pos[1:])
        neg = ' '.join(neg[1:])
        data['id'].append(id-1)
        data['failure'].append(failure)
        data['pos'].append(pos)
        data['neg'].append(neg)

    train_data, val_data = {'id': [], 'failure': [], 'pos': [], 'neg': []}, {'id': [], 'failure': [], 'pos': [], 'neg': []}
    train_indices = random.sample(list(range(len(data['id']))), k=int(len(data['id'])*0.9))
    for i in range(len(data['id'])):
        if i in train_indices:
            train_data['id'].append(data['id'][i]), train_data['failure'].append(data['failure'][i]), train_data['pos'].append(data['pos'][i]), train_data['neg'].append(data['neg'][i])
        else:
            val_data['id'].append(data['id'][i]), val_data['failure'].append(data['failure'][i]), val_data['pos'].append(data['pos'][i]), val_data['neg'].append(data['neg'][i])

    train_dataset = Dataset.from_dict(train_data)
    validation_dataset = Dataset.from_dict(val_data)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })

    dataset_dict.save_to_disk('data/dpo_react_data_datasetdict')

    train_df, val_df = pd.DataFrame(train_data), pd.DataFrame(val_data)
    train_df.to_csv('data/dpo_react_data_train.csv', index=False)
    val_df.to_csv('data/dpo_react_data_val.csv', index=False)


def sft_data_generation():
    data_path = 'data/positive_data'
    with open(data_path, 'r') as f:
         lines = f.readlines()
    
    id = 0
    data = {
        'id': [],
        'failure': [],
        'inference': []
    }

    for line in lines:
        line = eval(line)
        failure, inference = line[0], ' '.join(line[1:])
        data['id'].append(id)
        data['failure'].append(failure)
        data['inference'].append(inference)
    
    train_data, val_data = {'failure': [], 'inference': []}, {'failure': [], 'inference': []}
    train_indices = random.sample(list(range(len(data['id']))), k=int(len(data['id'])*0.9))
    for i in range(len(data['id'])):
        if i in train_indices:
            train_data['failure'].append(data['failure'][i]), train_data['inference'].append(data['inference'][i])
        else:
            val_data['failure'].append(data['failure'][i]), val_data['inference'].append(data['inference'][i])
    
    train_dataset = Dataset.from_dict(train_data)
    validation_dataset = Dataset.from_dict(val_data)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset
    })

    dataset_dict.save_to_disk('data/sft_react_data_datasetdict')
    train_df, val_df = pd.DataFrame(train_data), pd.DataFrame(val_data)
    train_df.to_csv('data/sft_react_data_train.csv', index=False)
    val_df.to_csv('data/sft_react_data_val.csv', index=False)


if __name__ == '__main__':
    dpo_data_generation()