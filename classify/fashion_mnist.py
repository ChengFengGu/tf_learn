#%%

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# %%
tf.__version__

#%%
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#%%
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
# plt.grid()
plt.show()

#%%
train_images = train_images / 255.0
test_images = test_images / 255.0


#%% [markdown]

# ---

# 下面可视化训练数据

# ---

#%%
plt.figure(figsize=(10, 10))
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()


#%% [markdown]
# ---
# 从现在开始，进入重头戏。开始构建网络模型了。
#
# ---

#%%

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10),
    ]
)

#%% [markdown]
# ---
# 以上代码其实就是传统的线性分类，将所有的像素拉伸成$（28\times 28）\times 1$，然后之后再进行全连接形成权重矩阵W，最后得出10个值。不出意外，下面会使用softmax分类。
#
# ---

#%%
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


#%% [markdown]
# ---
# 以上的代码就是模型编译，现在不太明白的是：这个`adam`和`稀疏分类交叉熵`是什么意思。
#
# 下面开始训练模型
# ---

#%%
model.fit(train_images, train_labels, epochs=10)

#%% [markdown]
# ---
# 下面开始评估模型
#
# ---

#%%
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)


#%% [markdown]
# ---
# 下面开始模型预测
#
# ---

#%%
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
#%%
class_names[np.argmax(predictions[0])], class_names[train_labels[0]]

#%%
#%% [markdown]
# ---
# 以上，模型就完成了全过程。下面我们可视化一下
#
# ---

#%%
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            class_names[predicted_label],
            100 * np.max(predictions_array),
            class_names[true_label],
        ),
        color=color,
    )


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


#%%

i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

#%%
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

#%%
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

#%%
img = test_images[0]
img = np.expand_dims(img,0)


#%%
predictions_single = probability_model.predict(img)

#%%
predictions_single

#%%
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

#%%
np.argmax(predictions_single[0])
