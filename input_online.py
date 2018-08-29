import pandas as pd
import numpy as np
import sys
import os


# 定义 topai()
def topai(i, result):
    print(result)


online = False

# 定义 model_dir
model_dir = "models/"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# 开始定义df1 ~ df4
df1 = pd.read_csv("input/data_tr.csv", encoding='utf-8')
df2 = pd.read_csv('input/w2v.csv', encoding='utf-8', header=None)
