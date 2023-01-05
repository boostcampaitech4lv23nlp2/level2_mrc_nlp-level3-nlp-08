import json
from collections import defaultdict

with open('./outputs/test_dataset_fold1_pretrain/nbest_predictions.json') as f:
    fold1 = json.load(f)
with open('./outputs/test_dataset_fold2_pretrain/nbest_predictions.json') as f:
    fold2 = json.load(f)
with open('./outputs/test_dataset_fold3_pretrain/nbest_predictions.json') as f:
    fold3 = json.load(f)
with open('./outputs/test_dataset_fold4_pretrain/nbest_predictions.json') as f:
    fold4 = json.load(f)
with open('./outputs/test_dataset_fold5_pretrain/nbest_predictions.json') as f:
    fold5 = json.load(f)

def most_frequent(data):
    return max(data, key=data.count)
    
mrc_id = fold1.keys()
mrc_id = list(mrc_id)

foldList = [fold1, fold2, fold3, fold4, fold5]

output = {}
for _id in mrc_id:
    dic = defaultdict()
    for fold in foldList:
        data = fold[_id]
        for d in data:
            try:
                dic[d['text']] += d['probability']
            except:
                dic[d['text']] = d['probability']
    sorted_dict = sorted(dic.items(), key=lambda item:item[1], reverse=True)
    answer = sorted_dict[0][0]

    output[_id] = answer

file_path = './kfold_pretrain_ensemble_predictions_soft.json'

with open(file_path, 'w') as out:
    json.dump(output, out, indent = 4, ensure_ascii=False)