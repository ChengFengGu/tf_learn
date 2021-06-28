#%%
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
import seaborn as sns
import json
import pathlib
from pathlib import Path
import tensorflow.keras as keras

#%%
dataset_path = keras.utils.get_file(
    "auto-mpg.data",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
)


#%%
dataset_path

#%% [markdown]
# ---
# .data文件可以直接用pandas读取。不得不说，牛皮
#
# ---

#%%
column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin",
]
raw_dataset = pd.read_csv(
    dataset_path,
    names=column_names,
    na_values="?",
    comment="\t",
    sep=" ",
    skipinitialspace=True,
)
dataset = raw_dataset.copy()
dataset.tail()

#%%
dataset.isna().sum()

#%%
dataset = dataset.dropna()

#%% [markdown]
# ---
# 以上代码，进行了数据清洗。下面就要进行编码的部分了
# 
# ---


#%%

