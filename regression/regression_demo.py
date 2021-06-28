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

Origin = dataset.pop("Origin")

#%% [markdown]
# ---
# 学到了，如果只要一行数据(不带header)，还可以用pop，牛皮。不过这个操作是转移，不是复制。所以可以用于标签和数据分离的时候。
#
# ---

#%%
origin = Origin

#%%
dataset["USA"] = (origin == 1) * 1.0
dataset["Europe"] = (origin == 2) * 1.0
dataset["Japan"] = (origin == 3) * 1.0

#%% [markdown]
# ---
# 其实以上部分只是独热码的另一个实现方式
#
# ---


#%%
dataset.tail()

#%% [markdown]
# ---
# 下面拆分数据集，分别用于训练和测试
#
# ---

#%%
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


#%%
train_dataset

#%%
sns.pairplot(
    train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde"
)

#%% [markdown]
# ---
# sns.pairplot是ML领域常见的应用，它可以直观的反映出各个features之间的线性关系
#
# ---


#%%
train_stats = train_dataset.describe()
train_stats

#%%
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

#%%
#%% [markdown]
# ---
# 以上代码可以查看各项数据的分布
#
# ---

#%%
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")

train_labels, test_labels

#%% [markdown]
# ---
# 以上代码分离出X,y
#
# ---

#%%
def norm(X):  # 定义标准化
    return X - train_stats["mean"] / train_stats["std"]


#%%
train_dataset

#%%
train_stats["mean"]

#%%
train_dataset - train_stats["mean"]

#%% [markdown]
# ---
# 看到这个操作了没，df之间也是可以直接进行四则运算的。牛皮。
#
# ---

#%%
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

#%% [markdown]
# ---
# 让我们的数据分布更合理，这个步骤有时也称为标准化
#
# ---

#%%


def build_model():
    model = keras.Sequential(
        [
            keras.layers.Dense(
                64, activation="relu", input_shape=[len(train_dataset.keys())]
            ),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1),
        ]
    )

    optimizer = keras.optimizers.RMSprop(0.01)

    model.compile(loss="mse", optimizer=optimizer, metrics=["mae", "mse"])

    return model


#%%
model = build_model()

#%%
model.summary()

#%%
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result

#%% [markdown]
# ---
# 为什么先不训练，就看结果呢。就是看一下模型的架构搭建是否正常
#
# ---

#%%
class printDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print(" ")
        print(".", end="")


EPOCHS = 1000

history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[printDot()],
)

#%% [markdown]
# ---
# 模型训练结束，下面可视化。
#
# ---

#%%
hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
hist.tail()

#%% [markdown]
# ---
# 可视化之路千万条，这种也很棒啊
#
# ---

#%%
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Abs Error, MAE")
    plt.plot(hist["epoch"], hist["mae"], label="Train Error")
    plt.plot(hist["epoch"], hist["val_mae"], label="Val Error")
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error [$MPG^2$]")
    plt.plot(hist["epoch"], hist["mse"], label="Train Error")
    plt.plot(hist["epoch"], hist["val_mse"], label="Val Error")
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


#%%
plot_history(history)

#%% [markdown]
# ---
# 可以看到，MAE和MSE的值还有慢慢升高的趋势，这不能被允许。那么，就让它在最低值停止折腾把。还可以继续利用回调。
#
# ---

#%%
model = build_model()
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=100)
history = model.fit(
    normed_train_data,
    train_labels,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[early_stop, printDot()],
)

plot_history(history)


#%% [markdown]
# ---
# 看见没，只有这个模型的验证指标100步没有提升，那就让它停下。看看，结果是不是好多了。
# 
# ---

#%%
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

#%% [markdown]
# ---
# 下面，我们就利用训练好的模型做个预测吧。
# 
# ---

test_predictions = model.predict(normed_test_data).flatten()

#%%
test_labels.shape

#%%

plt.scatter(test_labels,test_predictions)
plt.xlabel("True Value [MPG]")
plt.ylabel("Predictions [MPG]")
plt.axis('equal')
plt.axis('square')

plt.xlim([0,plt.ylim()[1]])
plt.ylim([0,plt.ylim()[1]])

_ = plt.plot([-100,100],[-100,100])

#%% [markdown]
# ---
# 看结果还行，上面的代码可视化了预测的结果。
# 
# ---