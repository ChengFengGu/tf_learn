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
# 以上的方法需要设置代理，具体可以查看：[tfds下载失败的解决方案](https://github.com/ChengFengGu/tf_learn/blob/master/docs/bugs/tfds_download_fail.md)
#
# ---

#%%

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

train_examples_batch


#%% [markdown]
# ---
# 可以看到，这个数据是纯粹的文本，没有经过任何预处理。
#
# ---

#%%

train_labels_batch

#%% [markdown]
# ---
# 构建网络
# - 词汇要嵌入
# - 模型有多少层
# - 多少隐层单元
#
# ---


#%% [markdown]
# ---
# 预训练文本嵌入模型。
# 常见的有几种：
# - [google/tf2-preview/gnews-swivel-20dim/1](https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1)预训练文本嵌入（text embedding）模型 。
# - [google/tf2-preview/gnews-swivel-20dim-with-oov/1](https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim-with-oov/1) ——类似 google/tf2-preview/gnews-swivel-20dim/1，但 2.5%的词汇转换为未登录词桶（OOV buckets）。如果任务的词汇与模型的词汇没有完全重叠，这将会有所帮助。
# - [google/tf2-preview/nnlm-en-dim50/1](https://hub.tensorflow.google.cn/google/tf2-preview/nnlm-en-dim50/1) ——一个拥有约 1M 词汇量且维度为 50 的更大的模型。
# - [google/tf2-preview/nnlm-en-dim128/1](https://hub.tensorflow.google.cn/google/tf2-preview/nnlm-en-dim128/1) ——拥有约 1M 词汇量且维度为128的更大的模型。
#
# ---

#%%
embedding = "https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

#%% [markdown]
# ---
# 以上代码，相当于我们跳过了学术界很麻烦的embeddding这个步骤，直接使用了现成的研究成果。
#
# ---

#%%
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(1))

#%%
model.summary()

#%% [markdown]
# ---
# 模型的架构搭建好了，现在开始定义损失函数和优化方法了。
#
# ---

#%%
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


#%% [markdown]
# ---
# 按照官方的说法，binaryCrossEntropy更适合处理概率--它适合度量概率之间的距离。
#
# ---


#%%
history = model.fit(
    train_data.shuffle(10000).batch(512),
    epochs=20,
    validation_data=validation_data.batch(512),
    verbose=1,
)

#%% [markdown]
# ---
# 以上代码我们成功的完成了训练的步骤。tfds支持data.shuffle(num).batch(b)
#
# ---

#%%

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s:%.3f" % (name, value))


#%%


