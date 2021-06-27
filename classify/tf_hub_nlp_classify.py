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
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

#%%
train_data,validation_data,test_data = tfds.load(
    name="imdb_reviews",
    split = ('train[:60%]','train[60%:]','test'),
    as_supervised = True
)
