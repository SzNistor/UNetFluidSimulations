"""
benchmark_unet.py

Generate predictions and plots of the test simulations.
The script was originally an .ipynb file used in Google Colab.

Author: Szilard Nistor
"""

"""
Specify the location of the network checkpoint and the simulation files used.
"""

network_folder = "UNet_BN_MSE"
network_number = 6

test_folder = "TEST_300"
train_folder = "TRAIN"

dimensions = (64,64,5)



"""
Basic Imports
"""
! pip install array2gif

import keras
import json
import glob
import time
import imageio
import cv2

import tensorflow as tf
import pandas as pd
import numpy as np

from array2gif import write_gif
from copy import copy
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity, variation_of_information
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

# Mount google drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Add folder containing simulations to path
import sys
sys.path.append("/content/drive/MyDrive/UNI/Bachelorarbeit")

from util.data import *
from util.path import *
from util.image import *


"""
Get Training Dataset Information
"""

means = pd.read_csv(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/{train_folder}/means.csv")['0'].values
stds = pd.read_csv(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/{train_folder}/stds.csv")['0'].values



"""
Get model weights
"""

network_path = f"/content/drive/MyDrive/UNI/Bachelorarbeit/network/{network_folder}/NET_{network_number:06d}"

with open(network_path + "/model.json") as f:
  model_info = json.load(f)
  model = tf.keras.models.model_from_json(json.dumps(model_info))

weight_files = sorted(glob.glob(network_path + "/cp*_final.h5"))
weight_file = weight_files[-1]

model.load_weights(weight_file)
print(f"Checkpoint loaded from {weight_file}")

"""
Get simulation files and generate paths
"""

sim_folders = sorted(glob.glob(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/{test_folder}/*"))[:]

prediction_path = f"/content/drive/MyDrive/UNI/Bachelorarbeit/predictions/{network_folder}/PRED"
prediction_path = get_unique_path(prediction_path)

make_dir(prediction_path)
make_dir(f"{prediction_path}/Average")

for sim_folder in sim_folders:
  test_simulation = sim_folder.split("/")[-1]
  make_dir(f"{prediction_path}/{test_simulation}")

"""
Read Simulation Files
"""

X = []
Y = []

for sim_folder in sim_folders:

  print(f"Reading files from: {sim_folder}")

  path = sim_folder + "/combined/"
  files = sorted(glob.glob(path + "*.npz"))

  sim_test_ids = [f for f in files if "label" not in f]
  sim_test_labels = {f : "scene_label".join(f.split("scene")) for f in files if "label" not in f}

  X_sim = np.empty((len(sim_test_ids), *dimensions))
  Y_sim = np.empty((len(sim_test_ids)+1, *dimensions))

  # Load each file (ordered) of the current simulation
  for i, ID in enumerate(sim_test_ids):
    x = load_scene(ID, means, stds, False, False, False)
    y = load_scene(sim_test_labels[ID], means, stds, False, False, False)

    X_sim[i,] = x
    # Add the first array to the simulation-array
    if i == 0:
      Y_sim[i,] = x
    
    Y_sim[i+1,] = y

    del x
    del y

  X.append(X_sim)
  Y.append(Y_sim)

  del X_sim
  del Y_sim

"""# Create Predictions"""

P = []
T = []

if network_folder == "UNet_BN_MSE" or network_folder == "UNet_BN_MSE_DET":

  for X_sim in X:
    x = X_sim[0]
    P_sim = [x]

    # Save the prediction and use it as an input for the following prediction
    start = time.time()
    for i in range(X_sim.shape[0]):
      x = model.predict(np.array([x]), verbose=False)[0]
      P_sim.append(x)

    end = time.time()
    prediction_time = end-start

    P.append(P_sim)

    T.append(prediction_time)

    del P_sim

elif network_folder == "UNet_BN_MSE_DET_010_STEPS":

  for X_sim in X:
    P_sim = []

    P_sim.extend(X_sim[:10])

    # Use the first 10 steps as a prediction basis in order to avoid empty spaces throghout the prediction
    x = X_sim[:10]

    # Save the predictions and use them as an input for the following prediction
    start = time.time()
    while len(P_sim) < X_sim.shape[0]:
      x = model.predict(x, verbose=False)
      P_sim.extend(x)

    end = time.time()
    prediction_time = end-start

    P.append(P_sim)

    T.append(prediction_time)

    del P_sim

elif network_folder == "UNet_BN_MSE_DET_020_STEPS":

  for X_sim in X:
    P_sim = []

    P_sim.extend(X_sim[:20])

    # Use the first 20 steps as a prediction basis in order to avoid empty spaces throghout the prediction
    x = X_sim[:20]

    # Save the predictions and use them as an input for the following prediction
    start = time.time()
    while len(P_sim) < X_sim.shape[0]:
      x = model.predict(x, verbose=False)
      P_sim.extend(x)

    end = time.time()
    prediction_time = end-start

    P.append(P_sim)

    T.append(prediction_time)

    del P_sim

elif network_folder == "UNet_BN_MSE_DET_CONST":

  for X_sim in X:

    # Save the first step as it contains the original flags and source
    x_first = X_sim[0]

    x = np.dstack([x_first[:, :, :2], x_first[:,:,1:]])

    P_sim = [x_first]
    start = time.time()

    # Save the prediction and use it as an input for the following prediction
    for i in range(X_sim.shape[0]):
      x = model.predict(np.array([x]), verbose=False)[0]
      P_sim.append(np.dstack([x_first[:, :, :1], x]))
      # Extend each predicted step with the original flags and source for the next prediction
      x = np.dstack([x_first[:, :, :2], x])

    end = time.time()
    prediction_time = end-start

    P.append(P_sim)

    T.append(prediction_time)

    del P_sim

elif network_folder == "UNet_BN_MSE_DET_CONST_010_STEPS":

  for X_sim in X:

    # Save the first step as it contains the original flags and source
    x_first = X_sim[0]

    P_sim = []

    # Use the first 10 steps as a prediction basis in order to avoid empty spaces throghout the prediction
    P_sim.extend(X_sim[:10])

    x = np.array([np.dstack([x_first[:, :, :2], pred[:,:,1:]]) for pred in P_sim])

    # Save the predictions and use them as an input for the following prediction
    start = time.time()
    while len(P_sim) < X_sim.shape[0]:
      x = model.predict(x, verbose=False)
      P_sim.extend([np.dstack([x_first[:, :, 0], pred]) for pred in x])
      # Extend each predicted step with the original flags and source for the next prediction
      x = np.array([np.dstack([x_first[:, :, :2], pred]) for pred in x])
    end = time.time()
    prediction_time = end-start

    P.append(P_sim[:X_sim.shape[0]+1])
    T.append(prediction_time)

    del P_sim

print(f"Average prediction time for {len(X_sim)+1} steps: {np.mean(T)}")
"""
Create GIFs
"""

for i, sim_folder in enumerate(copy(sim_folders)):
  test_simulation = sim_folder.split("/")[-1]
  make_dir(f"{prediction_path}/{test_simulation}/STEPS")

  # Read the density channel of each simulation and normalize the values between 0 and 255
  # Rotate the images and resize them for better viewing
  # Finally repeat the channel to cretae an RGB image
  IMG = [np.uint8(255*(I[:, :, 1]-np.min(I[:, :, 1])) / (np.max(I[:, :, 1])-np.min(I[:, :, 1]))) for I in Y[i]]
  IMG = [np.rot90(I, k=3) for I in IMG]
  IMG = [cv2.resize(I, dsize=(128, 128), interpolation=cv2.INTER_CUBIC) for I in IMG]
  IMG = [np.repeat(I[:, :, np.newaxis], 3, axis=2) for I in IMG]

  # Repeat for the prediction
  PRED_IMG = [np.uint8(255*(I[:, :, 1]-np.min(I[:, :, 1])) / (np.max(I[:, :, 1])-np.min(I[:, :, 1]))) for I in P[i]]
  PRED_IMG = [np.rot90(I, k=3) for I in PRED_IMG]
  PRED_IMG = [cv2.resize(I, dsize=(128, 128), interpolation=cv2.INTER_CUBIC) for I in PRED_IMG]
  PRED_IMG = [np.repeat(I[:, :, np.newaxis], 3, axis=2) for I in PRED_IMG]

  # Write GIFs
  write_gif(IMG, f"{prediction_path}/{test_simulation}/STEPS/density.gif", fps=20)
  write_gif(PRED_IMG, f"{prediction_path}/{test_simulation}/STEPS/density_predicted.gif", fps=20)


"""
Create Plots
"""

plt.rcParams.update({'font.size': 18})

# Density
channel = 1

# Display the development of the channel at the desired steps
for i, sim_folder in enumerate(copy(sim_folders)):

  test_simulation = sim_folder.split("/")[-1]
  make_dir(f"{prediction_path}/{test_simulation}/STEPS")

  # Save the initial state of the channel
  fig = plt.figure(figsize=(7, 7))  
  plt.imshow(np.rot90(Y[i][0].reshape(dimensions)[:, :, channel], k=2), cmap="gray")
  plt.title("Initial State \n")

  plt.savefig(f"{prediction_path}/{test_simulation}/STEPS/density_0.png")

  # Compare the prediction to the target at the desired steps and save the resulting image
  fig, axis = plt.subplots(2, 4, figsize=(9*3, 12))
  for j, step in enumerate([9, 49, 149, 299]):
    # Get min and max values for normalising of the plots
    v_max = max(np.max(Y[i][step][:, :, channel]), np.max(P[i][step][:, :, channel]))
    v_min = min(np.min(Y[i][step][:, :, channel]), np.min(P[i][step][:, :, channel]))
    mse = mean_squared_error(Y[i][step][:, :, channel], P[i][step][:, :, channel])
    ssim = structural_similarity(Y[i][step][:, :, channel], P[i][step][:, :, channel])
    I1 =axis[0][j].imshow(np.rot90(Y[i][step].reshape(dimensions)[:, :, channel], k=2), cmap="gray", vmax = v_max, vmin = v_min)
    axis[0][j].set_title(f"Step {step+1} \n")
    axis[1][j].imshow(np.rot90(P[i][step].reshape(dimensions)[:, :, channel], k=2), cmap="gray", vmax = v_max, vmin = v_min)
    axis[1][j].set_title(f"MSE: {mse:.2f}    SSIM: {ssim:.2f}  \n")
    fig.colorbar(I1, ax=axis[:, j], shrink=0.6)

  axis[0][0].set_ylabel("Mantaflow \n", rotation=90, size='large')
  axis[1][0].set_ylabel("Prediction \n", rotation=90, size='large')

  plt.savefig(f"{prediction_path}/{test_simulation}/STEPS/density_development.png")

# Flags
channel = 0

# Compare the first and last state of the flags channel
for i, sim_folder in enumerate(copy(sim_folders)):

  test_simulation = sim_folder.split("/")[-1]
  make_dir(f"{prediction_path}/{test_simulation}/STEPS")

  fig, axis = plt.subplots(2, 1, figsize=(8, 15))
  v_max = max(np.max(P[i][0][:, :, channel]), np.max(P[i][-1][:, :, channel]))
  v_min = min(np.min(P[i][0][:, :, channel]), np.min(P[i][-1][:, :, channel]))

  mse = mean_squared_error(P[i][0][:, :, channel], P[i][-1][:, :, channel])
  ssim = structural_similarity(P[i][0][:, :, channel], P[i][-1][:, :, channel])

  I1 =axis[0].imshow(np.rot90(P[i][0].reshape(dimensions)[:, :, channel], k=2), cmap="gray", vmax = v_max, vmin = v_min)
  axis[1].imshow(np.rot90(P[i][-1].reshape(dimensions)[:, :, channel], k=2), cmap="gray", vmax = v_max, vmin = v_min)
  axis[1].set_title(f"MSE: {mse:.2f}    SSIM: {ssim:.2f}  \n")
  fig.colorbar(I1, ax=axis[:], shrink=0.6)


  axis[0].set_ylabel("Initial State \n", rotation=90, size='large')
  axis[1].set_ylabel("Step 300 \n", rotation=90, size='large')

  plt.savefig(f"{prediction_path}/{test_simulation}/STEPS/flags_0_300.png", bbox="thight")

# Display the development of the channel at the desired steps
for i, sim_folder in enumerate(copy(sim_folders)):

  test_simulation = sim_folder.split("/")[-1]
  make_dir(f"{prediction_path}/{test_simulation}/STEPS")

  fig = plt.figure(figsize=(7, 7))  
  plt.imshow(np.rot90(Y[i][0].reshape(dimensions)[:, :, channel], k=2), cmap="gray")
  plt.title("Initial State \n")

  plt.savefig(f"{prediction_path}/{test_simulation}/STEPS/flags_0.png")

  fig, axis = plt.subplots(2, 4, figsize=(9*3, 12))
  for j, step in enumerate([9, 49, 149, 299]):
    v_max = max(np.max(Y[i][step][:, :, channel]), np.max(P[i][step][:, :, channel]))
    v_min = min(np.min(Y[i][step][:, :, channel]), np.min(P[i][step][:, :, channel]))
    mse = mean_squared_error(Y[i][step][:, :, channel], P[i][step][:, :, channel])
    ssim = structural_similarity(Y[i][step][:, :, channel], P[i][step][:, :, channel])
    I1 =axis[0][j].imshow(np.rot90(Y[i][step].reshape(dimensions)[:, :, channel], k=2), cmap="gray", vmax = v_max, vmin = v_min)
    axis[1][j].imshow(np.rot90(P[i][step].reshape(dimensions)[:, :, channel], k=2), cmap="gray", vmax = v_max, vmin = v_min)
    axis[0][j].set_title(f"Step {step+1} \n")
    axis[1][j].set_title(f"MSE: {mse:.2f}    SSIM: {ssim:.2f}  \n")
    fig.colorbar(I1, ax=axis[:, j], shrink=0.6)

  axis[0][0].set_ylabel("Mantaflow \n", rotation=90, size='large')
  axis[1][0].set_ylabel("Prediction \n", rotation=90, size='large')

  plt.savefig(f"{prediction_path}/{test_simulation}/STEPS/flags_development.png")

# Pressure
channel = 2

# Display the development of the channel at the desired steps
for i, sim_folder in enumerate(copy(sim_folders)):

  test_simulation = sim_folder.split("/")[-1]
  make_dir(f"{prediction_path}/{test_simulation}/STEPS")

  fig = plt.figure(figsize=(7, 7))  
  plt.imshow(np.rot90(Y[i][0].reshape(dimensions)[:, :, channel], k=2), cmap="plasma")
  plt.title("Initial State \n")

  plt.savefig(f"{prediction_path}/{test_simulation}/STEPS/pressure_0.png")

  fig, axis = plt.subplots(2, 4, figsize=(9*3, 12))
  for j, step in enumerate([9, 49, 149, 299]):
    v_max = max(np.max(Y[i][step][:, :, channel]), np.max(P[i][step][:, :, channel]))
    v_min = min(np.min(Y[i][step][:, :, channel]), np.min(P[i][step][:, :, channel]))
    mse = mean_squared_error(Y[i][step][:, :, channel], P[i][step][:, :, channel])
    ssim = structural_similarity(Y[i][step][:, :, channel], P[i][step][:, :, channel])
    I1 =axis[0][j].imshow(np.rot90(Y[i][step].reshape(dimensions)[:, :, channel], k=2), cmap="plasma", vmax = v_max, vmin = v_min)
    axis[1][j].imshow(np.rot90(P[i][step].reshape(dimensions)[:, :, channel], k=2), cmap="plasma", vmax = v_max, vmin = v_min)
    axis[0][j].set_title(f"Step {step+1} \n")
    axis[1][j].set_title(f"MSE: {mse:.2f}    SSIM: {ssim:.2f}  \n")
    fig.colorbar(I1, ax=axis[:, j], shrink=0.6)

  axis[0][0].set_ylabel("Mantaflow \n", rotation=90, size='large')
  axis[1][0].set_ylabel("Prediction \n", rotation=90, size='large')

  plt.savefig(f"{prediction_path}/{test_simulation}/STEPS/pressure_development.png")


# Velocity
# Display the development of the channel at the desired steps
for i, sim_folder in enumerate(copy(sim_folders)):

  test_simulation = sim_folder.split("/")[-1]
  make_dir(f"{prediction_path}/{test_simulation}/STEPS")

  fig = plt.figure(figsize=(7, 7))  
  plt.imshow(np.rot90(np.linalg.norm(Y[i][0][:, :, 3:4], axis=2), k=2), cmap="plasma")
  plt.title("Initial State \n")

  plt.savefig(f"{prediction_path}/{test_simulation}/STEPS/velocity_magnitude_0.png")

  fig, axis = plt.subplots(2, 4, figsize=(9*3, 12))
  for j, step in enumerate([9, 49, 149, 299]):

    vel_mag_p = np.linalg.norm(P[i][step][:, :, 3:4], axis=2)
    vel_mag_y = np.linalg.norm(Y[i][step][:, :, 3:4], axis=2)

    v_max = max(np.max(vel_mag_y), np.max(vel_mag_p))
    v_min = min(np.min(vel_mag_y), np.min(vel_mag_p))
    mse = mean_squared_error(vel_mag_y, vel_mag_p)
    ssim = structural_similarity(vel_mag_y, vel_mag_p)
    I1 =axis[0][j].imshow(np.rot90(vel_mag_y, k=2), cmap="plasma", vmax = v_max, vmin = v_min)
    axis[1][j].imshow(np.rot90(vel_mag_p, k=2), cmap="plasma", vmax = v_max, vmin = v_min)
    axis[0][j].set_title(f"Step {step+1} \n")
    axis[1][j].set_title(f"MSE: {mse:.2f}    SSIM: {ssim:.2f}  \n")
    fig.colorbar(I1, ax=axis[:, j], shrink=0.6)

  axis[0][0].set_ylabel("Mantaflow \n", rotation=90, size='large')
  axis[1][0].set_ylabel("Prediction \n", rotation=90, size='large')

  plt.savefig(f"{prediction_path}/{test_simulation}/STEPS/velocity_magnitude_development.png")