import json

with open('./outputs/test_dataset_fold1/nbest_predictions.json') as f:
    fold1 = json.load(f)
with open('./outputs/test_dataset_fold2/nbest_predictions.json') as f:
    fold2 = json.load(f)
with open('./outputs/test_dataset_fold3/nbest_predictions.json') as f:
    fold3 = json.load(f)
with open('./outputs/test_dataset_fold4/nbest_predictions.json') as f:
    fold4 = json.load(f)
with open('./outputs/test_dataset_fold5/nbest_predictions.json') as f:
    fold5 = json.load(f)

def most_frequent(data):
    return max(data, key=data.count)
    
mrc_id = fold1.keys()
mrc_id = list(mrc_id)

data = {}

for _id in mrc_id:
    tmp = [fold1[_id][0]['text'], fold2[_id][0]['text'], fold3[_id][0]['text'], fold4[_id][0]['text'], fold5[_id][0]['text']]
    data[_id] = most_frequent(tmp)

file_path = './kfold_ensemble_predictions.json'

with open(file_path, 'w') as out:
    json.dump(data, out, indent = 4, ensure_ascii=False)