#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#%% [markdown]
# ---
# !conda install tensorflow-hub
# !pip  install tfds-nightly
#
# ---

#%%
import tensorflow_hub as hub
import tensorflow_datasets as tfds

#%%

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print(
    "GPU is",
    "available"
    if tf.config.experimental.list_physical_devices("GPU")
    else "NOT AVAILABLE",
)

#%%
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=("train[:60%]", "train[60%:]", "test"),
    as_supervised=True,
)

#%% 
import os 
os.getcwd()

#%% [markdown]
# ---
# 以上的方法需要设置代理，具体可以查看：[tfds下载失败的解决方案](../docs/bugs/tfds_download_fail.md)
# 
# ---

#%%

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch