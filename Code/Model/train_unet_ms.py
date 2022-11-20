"""
train_unet_ms.py

Script for training the UNet on the data generated with Mantaflow. 
The script was originally an .ipynb file used in Google Colab.

Author: Szilard Nistor
"""

batch_size = 150
epochs = 100

network_nr = -1
steps = 1
"""
Basic Imports
"""

import glob
import time
import pathlib

import pandas as pd
import multiprocessing as mp
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
from IPython.core.display import clear_output
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPool2D, Conv2DTranspose, Input, concatenate, BatchNormalization, ReLU, Flatten
from keras.losses import MeanSquaredError

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Add folder containing datasets to path
import sys
sys.path.append("/content/drive/MyDrive/UNI/Bachelorarbeit")

from util.path import *
from util.image import *

"""
Check if GPU/TPU is connected
"""

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)

"""
Dataset Information
"""

training_dataset = "TRAIN_DET"
validation_dataset = "VAL_DET"

"""
Get Dataset Means and Standard Deviations
"""

means = pd.read_csv(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/{training_dataset}/means.csv")["0"].values
stds = pd.read_csv(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/{training_dataset}/stds.csv")["0"].values

"""
Get Training and Validation Ids
"""

train_simulations = glob.glob(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/{training_dataset}/*")

def get_step_label(f):
  global steps

  f_split = f.split("scene_")
  nr = int(f_split[-1].replace(".npz", ""))
  new_nr = nr + steps - 1

  if new_nr < 299:
    return "scene_label_".join([f_split[0], f"{new_nr:06d}.npz"])

  return None

def get_sim_scenes(sim):
  files = glob.glob(f"{sim}/combined/*.npz")
  
  sim_ids = [f for f in files if "label" not in f and get_step_label(f) is not None]
  sim_labels = {f : get_step_label(f) for f in files if "label" not in f and get_step_label(f) is not None}

  return sim_ids, sim_labels

def save_sim_scenes_training(result):
  global train_ids
  global train_labels

  train_ids.extend(result[0])
  train_labels.update(result[1])

def save_sim_scenes_validation(result):
  global val_ids
  global val_labels

  val_ids.extend(result[0])
  val_labels.update(result[1])

train_ids = []
train_labels = {}

pool = mp.Pool(32)
results = []

train_simulations = glob.glob(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/{training_dataset}/*")

for sim in train_simulations:
  r = pool.apply_async(get_sim_scenes, (sim,), callback=save_sim_scenes_training)
  results.append(r)

for i, r in enumerate(results):
  clear_output(wait=True)
  print(f"Gathering Training Ids: {(i+1)/(len(train_simulations))*100:.2f} %")
  r.wait()

pool.close()
pool.join()

print(f"{len(train_ids)}")

val_ids = []
val_labels = {}

pool = mp.Pool(32)
results = []

val_simulations = glob.glob(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/{validation_dataset}/*")

for sim in val_simulations:
  r = pool.apply_async(get_sim_scenes, (sim,), callback=save_sim_scenes_validation)
  results.append(r)

for i, r in enumerate(results):
  clear_output(wait=True)
  print(f"Gathering Validation Ids: {(i+1)/(len(val_simulations))*100:.2f} %")
  r.wait()

pool.close()
pool.join()
print(f"{len(val_ids)}")

"""
Data Generator
"""

# Based on a tutorial provided by Afshine Amidi and Shervine Amidi (https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)
class DeepFlowDataGen(keras.utils.Sequence):
  
  def __init__(self, 
               ids, 
               labels, 
               means=np.zeros(6), 
               stds=np.ones(6),
               batch_size=300, 
               dim=(64,64), 
               n_channels=5, 
               shuffle=True):

    self.ids = ids
    self.labels = labels

    self.means = means
    self.stds = stds

    self.batch_size = batch_size
    self.dim = dim
    self.n_channels = n_channels
    self.shuffle = shuffle

    self.on_epoch_end()

  def on_epoch_end(self):
    self.indexes = np.arange(len(self.ids))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def remap_obstacles(self, arr):
    # Remap Flags to 0 and 1
    s = arr[:, :, 0]
    # Set empty spaces to 0
    s[s == 1065353216.0] = 0
    # Set borders to 1
    s[s == 1101004800.0 ] = 1
    # Set obstacles and walls to 2
    s[s == 1073741824.0 ] = 2
    arr[:, :, 0] = s
    return arr

  # Use the means and stds of the velocity magnitude when normalizing
  def normalize(self, arr):
    self.means[3], self.means[4] = self.means[5], self.means[5]
    self.stds[3], self.stds[4] = self.stds[5], self.stds[5]

    for i in range(arr.shape[2]):
      arr[:, :, i] -= self.means[i]
      arr[:, :, i] *= (1/self.stds[i])

    return arr

  def __data_generation(self, batch_ids):
    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    Y = np.empty((self.batch_size, *self.dim, self.n_channels))

    # Generate data
    for i, ID in enumerate(batch_ids):
      x = np.load(ID)["arr_0"]
      y = np.load(self.labels[ID])["arr_0"]

      # Remap obstacle flags to 1 
      x = self.remap_obstacles(x)
      y = self.remap_obstacles(y)

      # Normalize Data
      x = self.normalize(x)
      y = self.normalize(y)

      X[i,] = x
      Y[i,] = y

      del x
      del y

    return X, Y

  

  def __len__(self):
    return int(np.floor(len(self.ids) / self.batch_size))

  def __getitem__(self, index):
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    batch_ids = [self.ids[k] for k in indexes]

    X, Y = self.__data_generation(batch_ids)

    return X, Y

params = {
    "means": means,
    "stds": stds,
    "batch_size" : batch_size, 
    "dim" : (64,64), 
    "n_channels" : 5, 
    "shuffle" : True
}

training_generator = DeepFlowDataGen(train_ids, train_labels, **params)
validation_generator = DeepFlowDataGen(val_ids, val_labels, **params)

"""
Model Information
"""

network_path = f"/content/drive/MyDrive/UNI/Bachelorarbeit/network/UNet_BN_MSE_DET_{steps:03d}_STEPS"

p = pathlib.Path(network_path)
p.mkdir(parents=True, exist_ok=True)

if network_nr != -1:
  network_path += f"/NET_{network_nr:06d}"
else:
  network_path = get_unique_path(network_path + "/NET")
  make_dir(network_path)

print(f"Network Path: {network_path}")

"""
Define Model
"""

from tensorflow.keras import layers
from tensorflow.keras import activations

# Based on the tutorial provided by Margaret Maynard-Reid (https://pyimg.co/6m5br)
def double_conv_block(x, n_filters):
   # Conv2D + BN + ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same")(x)#, kernel_initializer = "he_normal")(x)
   x = layers.BatchNormalization()(x)
   x = layers.ReLU()(x)
   # Conv2D + BN + ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same")(x)#, kernel_initializer = "he_normal")(x)
   x = layers.BatchNormalization()(x)
   x = layers.ReLU()(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x

def build_unet_model():
  # inputs
  inputs = layers.Input(shape=(64,64,5))
  # encoder: contracting path - downsample
  # 1 - downsample
  f1, p1 = downsample_block(inputs, 64)
  # 2 - downsample
  f2, p2 = downsample_block(p1, 128)
  # 3 - downsample
  f3, p3 = downsample_block(p2, 256)
  # 4 - downsample
  f4, p4 = downsample_block(p3, 512)
  # 5 - bottleneck
  bottleneck = double_conv_block(p4, 1024)
  # decoder: expanding path - upsample
  # 6 - upsample
  u6 = upsample_block(bottleneck, f4, 512)
  # 7 - upsample
  u7 = upsample_block(u6, f3, 256)
  # 8 - upsample
  u8 = upsample_block(u7, f2, 128)
  # 9 - upsample
  u9 = upsample_block(u8, f1, 64)
  # outputs
  outputs = layers.Conv2D(5, 1, padding="same", activation = "linear")(u9)
  # unet model with Keras Functional API
  unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
  return unet_model

model = build_unet_model()
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

"""## Model Chekpoints"""

checkpoint_files = sorted(glob.glob(network_path + "/cp*.ckpt.index"))
checkpoint_path = network_path + f"/cp_{len(checkpoint_files):06d}.ckpt"
print(f"Checkpoint Path: {checkpoint_path}")

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                               save_weights_only=True,
                                                               save_freq="epoch",
                                                               save_best_only=True,
                                                               monitor='val_loss',
                                                               mode='min'
                                                               )

history_files = glob.glob(network_path + "/history*.csv")
history_path = network_path + f"/history_{len(history_files):06d}.csv"
print(f"History Path: {history_path}")


model_history_callback = keras.callbacks.CSVLogger(history_path)

# Write model architecture to json so it can be loaded later
if network_nr == -1:
  model_json = model.to_json()
  with open(network_path + "/model.json", "w") as json_file:
    json_file.write(model_json)

# Load weights if existent
if network_nr != -1:
  weight_files = sorted(glob.glob(network_path + "/cp*.ckpt.index"))
  weight_file = weight_files[-1].replace(".index", "")

  model.load_weights(weight_file)

  print(f"Checkpoint loaded from {weight_file}")

"""
Train Model
"""

start = time.time()

training_history = model.fit(x=training_generator,
                             validation_data=validation_generator,
                             epochs=epochs,
                             callbacks=[model_checkpoint_callback, model_history_callback],
                             use_multiprocessing=False,
                             workers=16)

end = time.time()
training_time = end - start

print("")
print(f"Training finished after {training_time} seconds")

model.save_weights(checkpoint_path.replace(".ckpt", "_final.h5"))

"""
Write and Save the Training Information
"""

import json
training_summary = {
        "training_time": str(training_time // 60) + " m",
        "train": {
            "accuracy": np.mean(training_history.history['accuracy']),
            "loss": np.mean(training_history.history['loss'])
        },
        "validation": {
            "accuracy": np.mean(training_history.history['val_accuracy']),
            "loss": np.mean(training_history.history['val_loss'])
        }
    }
 
with open(network_path + "/summary_new.json", "w") as outfile:
  json.dump(training_summary, outfile)

plt.plot(training_history.history['accuracy'])
plt.plot(training_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(network_path + "/model_accuracy_new.png")
#plt.clf()

plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(network_path + "/model_loss_new.png")
#plt.clf()

pd.DataFrame(training_history.history).to_csv(history_path, index=False)

