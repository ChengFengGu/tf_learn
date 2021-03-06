#%%
import numpy as np
from pandas.core.algorithms import mode
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)
import logging

logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

#%% [markdown]
# ---
# # 1. 准备数据集
#
# ---


#%%

logging.info("preparing imdb dataset")

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data()

#%%
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# %%
logging.info(f"train_data[0]==>\n{train_data[0]}")

#%%

logging.info(f"训练数据中各个评论的长度并不一致==>")
logging.info(
    f"len of train[0]:{len(train_data[0])} || len of train[1]:{len(train_data[1])}"
)

#%% [markdown]
# ---
# 以上代码，我们探索了数据。这个数据是评论文本，每个数字代表了一个单词，那么，怎样才能将数字转化为单词呢？
#
# ---

#%%

word_index = imdb.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

reverse_word_index

#%% [markdown]
# ---
# 卧槽，这个代码非常有意思，值得学习。以后两列的变换除了pandas，这种方式也会场的方便啊。
#
# ---

#%%
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


#%%
decode_review(train_data[0])

#%% [markdown]
# ---
# # 正式流程
#
# ---

#%% [markdown]
# ---
# 为了确保输入数据的长度一致，我们必须做一些改进，可以填充数字为最长的文本长度。keras提供了`pad_sequece`函数来进行这个操作
#
# ---

#%%
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding="post", maxlen=256
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index["<PAD>"], padding="post", maxlen=256
)


#%%
len(train_data[0]), len(train_data[1])


#%% [markdown]
# ---
# 用于NLP的神经网络与CV领域的有一些不同，这里仔细看一下。
#
# ---

#%% [markdown]
# ---
# [vocab size 应该在官网教程的基础上改一下](https://github.com/TannerGilbert/Tutorials/issues/1#issuecomment-480898786)
#
#
# ---

#%%
import pandas as pd

df = pd.DataFrame(word_index, index=[1]).T
df

#%%
max(df.values)[0]


#%%
vocab_size = max(df.values)[0] + 1  # 词汇数目

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

#%% [markdown]
# ---
# 模型已经构建完毕，下面开始编译模型。在keras中，编译模型需要指定的内容包括损失函数，优化方式以及评估方法
#
# ---

#%%
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#%%

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

#%% [markdown]
# ---
# 上述代码创建了一个验证集合
#
# ---
#%%

# partial_x_train = np.expand_dims(partial_x_train,axis=0)
# partial_y_train = np.expand_dims(partial_y_train,axis=0)

# x_val = np.expand_dims(x_val,axis=0)
# y_val = np.expand_dims(y_val,axis=0)

#%%
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=40,
    batch_size=512,
    validation_data=(x_val, y_val),
    verbose=1,
)


#%% [markdown]
# ---
# # 评估模型
#
# ---


#%% [markdown]
# ---
# - verbose = 0 为不在标准输出流输出日志信息
# - verbose = 1 为输出进度条记录
# - verbose = 2 为每个epoch输出一行记录
#
# ---


#%%

results = model.evaluate(test_data, test_labels, verbose=2)
results

#%% [markdown]
# ---
# # 可视化
#
# ---

#%%
history_dict = history.history
history_dict.keys()

#%%
import matplotlib.pyplot as plt

epochs = range(1, len(history_dict["accuracy"]) + 1)

plt.plot(epochs, history_dict["loss"], "bo", label="Traing loss")
plt.plot(epochs, history_dict["val_loss"],"b",label="Validation loss")
plt.title('Traing and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#%%

plt.clf()

plt.plot(epochs,history_dict['val_accuracy'],'bo',label="val accuracy")
plt.plot(epochs, history_dict['accuracy'],'b',label="train accuracy")
plt.title("train and validation accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()




