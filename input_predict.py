import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import json

# 定义 model_dir
model_dir = "models/"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# 开始定义df1 ~ df4
df1 = pd.read_csv("input/data_te.csv", encoding='utf-8')
label1 = df1['label']
df1 = df1.drop(['label'], axis=1)
result_path = 'experimental_result.json'


def update_json(path, data):
    if os.path.exists(path):
        with open(path, 'r') as f:
            data_o = json.load(f)
        data_o.update(data)
        data = data_o
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)


# 定义 topai()
def topai(i, result):
    if i == 1:
        assert result.shape[0] == df1.shape[0]
        assert result.shape[1] == 2
        t = result['label']
        assert t.dtype == np.int32
        assert t.max() <= 1
        assert t.min() >= 0
        # print(result)
        s = f1_score(label1, t)
        print(s)
        update_json(result_path, {'f1': s})
    else:
        assert result.shape[0] == df1.shape[0]
        assert result.shape[1] == 2
        t = result['label']
        assert t.dtype == np.float32
        assert t.max() <= 1
        assert t.min() >= 0
        # print(result)
        s = (roc_auc_score(label1, t))
        print(s)
        update_json(result_path, {'auc': s})
        for th in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            ph = np.array(t > th, dtype=np.int32)
            s = f1_score(label1, ph)
            print(th, s)
            update_json(result_path, {'th%f' % th: s})
