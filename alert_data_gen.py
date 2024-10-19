import os
import json
import re

alerts_dir = "/home/lzq/react-rl/alerts-0731/alerts/EDOC1100372111"
alert_dict = {}

def parse_json(json_data:dict)->list:
    steps = json_data["处理步骤"]
    react_data = [{"next_step":1, "text":[], "ids":[], "end":False}]
    while True:
        flag = False
        for data in react_data:
            if data["end"] == True:
                flag = True
                break
        if flag:
            break

        for data in react_data:
            if data["end"] == True:
                continue
            find_step = None
            for step in steps:
                if step["step"] == data["next_step"]:
                    find_step = step
                    break
            text = find_step["text"]
            step_id = find_step["step"]
            if "links" in find_step:
                link_steps_id = []
                for link in find_step["links"]:
                    if "target_step" in link and link["target_step"] > step_id:
                        link_steps_id.append(link["target_step"])
                link_steps_id = list(set(link_steps_id))
                _data = data.copy()
                if len(link_steps_id) == 1:
                    data["next_step"], _data["next_step"] = None, link_steps_id[0]
                    data["text"].append(text), _data["text"].append(text), 
                    data["ids"].append(step_id), _data["ids"].append(step_id)
                    data["end"] = True
                    react_data.append(_data)
                elif len(link_steps_id) == 2:
                    data["next_step"], _data["next_step"] = link_steps_id[0], link_steps_id[1]
                    data["text"].append(text), _data["text"].append(text), 
                    data["ids"].append(step_id), _data["ids"].append(step_id)
                    react_data.append(_data)
                else:
                    if step_id == len(steps):
                        data["next_step"] = None
                        data["text"].append(text)
                        data["ids"].append(step_id)
                        data["end"] = True
                    else:
                        data["next_step"] = step_id+1
                        data["text"].append(text)
                        data["ids"].append(step_id)
            elif step_id == len(steps):
                data["next_step"] = None
                data["text"].append(text)
                data["ids"].append(step_id)
                data["end"] = True
            else:
                data["next_step"] = step_id+1
                data["text"].append(text)
                data["ids"].append(step_id)
    
    return react_data
                # pattern = re.compile(r"是，(.*?)。否，(.*?)")
                # match = pattern.match(text)
                # if match:
                #     target_text = match.group()
                # else:
                #     pass
                # branch_1, branch_2 = target_text.split('。')
                # _data = data.copy()
                # next_1, next_2 = None, None
                # for next_step_id in link_steps_id:
                #     if next_step_id in branch_1:
                #         next_1 = next_step_id
                #     if next_step_id in branch_2:
                #         next_2 = next_step_id
                # if next_1:
                #     data["next_step"] = next_1
                #     data["text"].append(text[:-len(target_text)]+branch_1)
                # else:
                #     data["next_step"] = None
                #     data["text"].append(text[:-len(target_text)]+branch_1)
                #     data["end"] = True
                # if next_2:
                #     _data["next_step"] = next_2
                #     _data["text"].append(text[:-len(target_text)]+branch_2)
                # else:
                #     _data["next_step"] = None
                #     _data["text"].append(text[:-len(target_text)]+branch_2)
                #     _data["end"] = False

              
def fill_alert_dict():
    fp = [os.path.join(alerts_dir, p) for p in os.listdir(alerts_dir)]
    fpp = []
    for fp_ in fp:
        fpp += [os.path.join(fp_, p) for p in os.listdir(fp_)]

    for item in fpp:
        if os.path.isdir(item):
            fppp = [os.path.join(item, p) for p in os.listdir(item)]
            for item_ in fppp:
                if item_.endswith('json'):
                    alert_dict[item_.split('/')[-2]+' '+item_.split('/')[-1]] = item_
        elif item.endswith('json'):
            alert_dict[item.split('/')[-1]] = item
        else:
            continue
        
    return alert_dict
    

alert_dict = fill_alert_dict()
print(len(alert_dict))
react_data = []
for alert, alert_path in alert_dict.items():
    with open(alert_path, 'r') as f:
        json_alert = json.loads(f.read())
        react_data += parse_json(json_alert)

print(react_data[0])