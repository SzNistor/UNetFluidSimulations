"""
get_signal.py

Generate signals to analyze. For each signal claculate the magnitude and power spectra. 
Finally save each signal, its error and relative error compared to the baseline simulation.

Author: Szilard Nistor
"""

"""
Specify the location of the network checkpoint and the simulation files used.
"""

network_folder = "UNet_BN_MSE_DET"
network_number = 1

baseline_folder = "TEST_VORTICITY_DET/VORTICITY_0"
train_folder = "TRAIN_DET"

dimensions = (64,64,5)
results_folder = "results_det"

"""
Basic Imports
"""

import keras
import json
import glob
import time
import gzip
import pickle

import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.stats as stats

from copy import copy
from matplotlib import pyplot as plt

# Mount google drive
from google.colab import drive
drive.mount('/content/drive')

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
Get Baseline Simulation Files
"""

sim_folders = sorted(glob.glob(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/{baseline_folder}/*")[:])

"""
Define Functions Needed For Analysis
"""

R = np.array([np.fft.fftfreq(64)] * 64)
FFT_M = np.sqrt(np.square(R) + np.square(R.T))

# Get the power of the signal at each step as a dictionary of frequency:[power at each step].
# Based on the tutorial provided by Bert Vandenbroucke (https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/)
def get_signal_power(mean_sim):
  kfreq = np.fft.fftfreq(64) * 64
  kfreq2D = np.meshgrid(kfreq, kfreq)
  knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

  knrm = knrm.flatten()

  kbins = np.arange(0.5, 33, 1.)
  kvals = 0.5 * (kbins[1:] + kbins[:-1])

  dict_out = {f: [] for f in kvals.flatten()}

  for i in range(len(mean_sim)):
    I = mean_sim[i][:, :, 1]
    I_FFT = np.fft.fft2(I)

    FFT_POW = np.abs(I_FFT)**2

    fourier_amplitudes = FFT_POW.flatten()

    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic = "mean",
                                     bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    
    for i, f in enumerate(kvals):
      dict_out[f].append(Abins[i])

  return dict_out

# Get the power of the residual of the signal at each step as a dictionary of frequency:[power at each step].
# Based on the tutorial provided by Bert Vandenbroucke (https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/)
def get_residual_power(mean_sim, baseline_sim):
  kfreq = np.fft.fftfreq(64) * 64
  kfreq2D = np.meshgrid(kfreq, kfreq)
  knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

  knrm = knrm.flatten()

  kbins = np.arange(0.5, 33, 1.)
  kvals = 0.5 * (kbins[1:] + kbins[:-1])

  dict_out = {f: [] for f in kvals.flatten()}

  for i in range(len(mean_sim)):
    I = mean_sim[i][:, :, 1] - baseline_sim[i][:, :, 1]
    I_FFT = np.fft.fft2(I)

    FFT_POW = np.abs(I_FFT)**2

    fourier_amplitudes = FFT_POW.flatten()

    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic = "mean",
                                     bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    
    for i, f in enumerate(kvals):
      dict_out[f].append(Abins[i])

  return dict_out

# Get the magnitude of the frequencies at each step as a dictionary of frequency:[magnitude at each step].
def get_signal_magnitude(mean_sim):
  global FFT_M

  dict_out = {f: [] for f in FFT_M.flatten()}

  for i in range(len(mean_sim)):
    I_D = {f: [] for f in FFT_M.flatten()}

    I = mean_sim[i][:, :, 1]
    I_FFT = np.fft.fft2(I)

    for x in range(64):
      for y in range(64):
        I_D[FFT_M[y][x]].append(np.abs(I_FFT[y][x]))

    I_D = {f : np.mean(v) for f,v in I_D.items()}
    for k,v in I_D.items():
      dict_out[k].append(v)
    
  dict_out = dict(sorted(dict_out.items()))

  return dict_out

# Get the magnitude of the residual at each step as a dictionary of frequency:[magnitude at each step].
def get_residual_magnitude(mean_sim, baseline_sim):
  global FFT_M

  dict_out = {f: [] for f in FFT_M.flatten()}

  for i in range(len(mean_sim)):
    I_D = {f: [] for f in FFT_M.flatten()}

    I = mean_sim[i][:, :, 1] - baseline_sim[i][:, :, 1]
    I_FFT = np.fft.fft2(I)

    for x in range(64):
      for y in range(64):
        I_D[FFT_M[y][x]].append(np.abs(I_FFT[y][x]))

    I_D = {f : np.mean(v) for f,v in I_D.items()}
    for k,v in I_D.items():
      dict_out[k].append(v)
    
  dict_out = dict(sorted(dict_out.items()))

  return dict_out

# Get the frequency bands x=0, x=32, y=0, y=32 for the density at each step as arrays.
def get_signal_bands(mean_sim):
  x_0 = []
  x_32 = []
  y_0 = []
  y_32 = []
  
  for i in range(len(mean_sim)):
    I = mean_sim[i][:, :, 1]
    I_FFT = np.fft.fftshift(np.fft.fft2(I))

    x_0.append(np.abs(I_FFT)[:, 0])
    x_32.append(np.abs(I_FFT)[:, 32])
    y_0.append(np.abs(I_FFT)[0, :])
    y_32.append(np.abs(I_FFT)[32, :])

  return x_0, x_32, y_0, y_32

# Get the frequency bands x=0, x=32, y=0, y=32 for the residual at each step as arrays.
def get_residual_bands(mean_sim, baseline_sim):
  x_0 = []
  x_32 = []
  y_0 = []
  y_32 = []
  
  for i in range(len(mean_sim)):
    I = mean_sim[i][:, :, 1] - baseline_sim[i][:, :, 1]
    I_FFT = np.fft.fftshift(np.fft.fft2(I))

    x_0.append(np.abs(I_FFT)[:, 0])
    x_32.append(np.abs(I_FFT)[:, 32])
    y_0.append(np.abs(I_FFT)[0, :])
    y_32.append(np.abs(I_FFT)[32, :])

  return x_0, x_32, y_0, y_32

# Given the signal, baseline signal and file name, save the magnitude spectra, power spectra and frequency bands of the signal, residual and relative error.

def create_files(sim, baseline_sim, file_name):
  with gzip.open(f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{file_name}.npy.gz', 'wb') as array_file:
    np.save(array_file, sim)
  
  sim_Magnitude = get_signal_magnitude(sim)

  with gzip.open(f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/SIGNAL/{file_name}-magnitude.pickle.gz', 'wb') as dict_file:
    pickle.dump(sim_Magnitude, dict_file)

  sim_Residual_Magnitude = get_residual_magnitude(sim, baseline_sim)
  
  with gzip.open(f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/RESIDUAL/{file_name}-magnitude.pickle.gz', 'wb') as dict_file:
    pickle.dump(sim_Residual_Magnitude, dict_file)

  sim_Relative_Error_Magnitude = {k: np.asarray(sim_Residual_Magnitude[k]) / np.asarray(sim_Magnitude[k]) for k in sim_Residual_Magnitude}
  
  with gzip.open(f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/RELATIVE_ERROR/{file_name}-magnitude.pickle.gz', 'wb') as dict_file:
    pickle.dump(sim_Relative_Error_Magnitude, dict_file)

  sim_Power = get_signal_power(sim)
  
  with gzip.open(f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/SIGNAL/{file_name}-power.pickle.gz', 'wb') as dict_file:
    pickle.dump(sim_Power, dict_file)

  sim_Residual_Power = get_residual_power(sim, baseline_sim)
  
  with gzip.open(f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/RESIDUAL/{file_name}-power.pickle.gz', 'wb') as dict_file:
    pickle.dump(sim_Residual_Power, dict_file)

  sim_Relative_Error_Power = {k: np.asarray(sim_Residual_Power[k]) / np.asarray(sim_Power[k]) for k in sim_Residual_Power}
  
  with gzip.open(f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/RELATIVE_ERROR/{file_name}-power.pickle.gz', 'wb') as dict_file:
    pickle.dump(sim_Relative_Error_Power, dict_file)

  sim_Bands = get_signal_bands(sim)
  sim_Residual_Bands = get_residual_bands(sim, baseline_sim)
  bands = ["x0", "x32", "y0", "y32"]

  for i in range(len(sim_Bands)):
    with gzip.open(f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/SIGNAL/{file_name}-band-{bands[i]}.pickle.gz', 'wb') as dict_file:
      pickle.dump(sim_Bands[i], dict_file)

    with gzip.open(f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/RESIDUAL/{file_name}-band-{bands[i]}.pickle.gz', 'wb') as dict_file:
      pickle.dump(sim_Residual_Bands[i], dict_file)

    with gzip.open(f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/RELATIVE_ERROR/{file_name}-band-{bands[i]}.pickle.gz', 'wb') as dict_file:
      pickle.dump(np.asarray(sim_Residual_Bands[i]) / np.asarray(sim_Bands[i]), dict_file)


"""
Apply the Function Defined to Multiple Signals
"""

"""
Vorticity 0.1 - Baseline
"""

# Read each simulation
X_B = []
Y_B = []

for sim_folder in sorted(sim_folders):

  print(f"Reading files from: {sim_folder}")

  path = sim_folder + "/combined/"
  files = sorted(glob.glob(path + "*.npz"))

  sim_test_ids = [f for f in files if "label" not in f]
  sim_test_labels = {f : "scene_label".join(f.split("scene")) for f in files if "label" not in f}

  X_sim = np.empty((len(sim_test_ids), *dimensions))
  Y_sim = np.empty((len(sim_test_ids)+1, *dimensions))


  for i, ID in enumerate(sim_test_ids):
    x = load_scene(ID, means, stds, False, False, False)
    y = load_scene(sim_test_labels[ID], means, stds, False, False, False)

    X_sim[i,] = x

    if i == 0:
      Y_sim[0,] = x

    Y_sim[i+1,] = y

    del x
    del y

  X_B.append(X_sim)
  Y_B.append(Y_sim)

  del X_sim
  del Y_sim

# Create the stepwise mean simulation
Y_B_M = np.mean(np.asarray(Y_B), axis=(0))

# Save the first image as a reference
with gzip.open(f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/start-band.pickle.gz', 'wb') as dict_file:
  pickle.dump(Y_B_M[0][:, :, 1], dict_file)

# Create necessary files
create_files(Y_B_M, Y_B_M, "vorticity-0_1")

"""
Further Vorticity Levels
"""

vorticity_folders = {
    "vorticity-0_101" : sorted(glob.glob(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/TEST_VORTICITY_DET/VORTICITY_0.001/*")),
    "vorticity-0_15" : sorted(glob.glob(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/TEST_VORTICITY_DET/VORTICITY_0.05/*")),
    "vorticity-0_3" : sorted(glob.glob(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/TEST_VORTICITY_DET/VORTICITY_0.2/*")),
    "vorticity-0_5" : sorted(glob.glob(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/TEST_VORTICITY_DET/VORTICITY_0.4/*")),
    "vorticity-1" : sorted(glob.glob(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/TEST_VORTICITY_DET/VORTICITY_0.9/*")),
}

vorticity_simulations = {k: [] for k in vorticity_folders}

# For each vorticity level read the simulations
for v, f in vorticity_folders.items():
  print(v)

  for sim_folder in f:

    path = sim_folder + "/combined/"
    files = sorted(glob.glob(path + "*.npz"))


    sim_test_ids = [f for f in files if "label" not in f]
    sim_test_labels = {f : "scene_label".join(f.split("scene")) for f in files if "label" not in f}

    print(f"Reading files from: {sim_folder} - {len(sim_test_ids)}")

    Y_sim = np.empty((len(sim_test_ids)+1, *dimensions))


    for i, ID in enumerate(sim_test_ids):
      y = load_scene(sim_test_labels[ID], means, stds, False, False, False)

      if i == 0:
        x = load_scene(ID, means, stds, False, False, False)
        Y_sim[0,] = x

      Y_sim[i+1,] = y

      del y

    vorticity_simulations[v].append(Y_sim)

    del Y_sim

# For each vorticity level determine the step-wise mean simulation.
for k, v in vorticity_simulations.items():
  mean_SIM = np.mean(np.asarray(v), axis=(0))
  create_files(mean_SIM, Y_B_M, k)

"""
UNet 1 Step
"""

# Load model weights
network_path = f"/content/drive/MyDrive/UNI/Bachelorarbeit/network/UNet_BN_MSE_DET/NET_000001"

with open(network_path + "/model.json") as f:
  model_info = json.load(f)
  model = tf.keras.models.model_from_json(json.dumps(model_info))

weight_files = sorted(glob.glob(network_path + "/cp*_final.h5"))
weight_file = weight_files[-1]

model.load_weights(weight_file)
print(f"Checkpoint loaded from {weight_file}")


# Create predictions
P = []
T = []

s = 0

for X_sim in X_B:

  print(sim_folders[s])
  s+=1

  x = X_sim[0]

  P_sim = [x]

  for i in range(X_sim.shape[0]):
    x = model.predict(np.array([x]), verbose=False)[0]
    P_sim.append(x)

  P.append(P_sim)
  del P_sim

# Create step-wise mean prediction
Y_P_M = np.mean(np.asarray(P), axis=(0))
create_files(Y_P_M, Y_B_M, "prediction-01-step")

"""
UNet 1 Step - Constant
"""

# Load model weights
network_path = f"/content/drive/MyDrive/UNI/Bachelorarbeit/network/UNet_BN_MSE_DET/NET_000006"

with open(network_path + "/model.json") as f:
  model_info = json.load(f)
  model = tf.keras.models.model_from_json(json.dumps(model_info))

weight_files = sorted(glob.glob(network_path + "/cp*_final.h5"))
weight_file = weight_files[-1]

model.load_weights(weight_file)
print(f"Checkpoint loaded from {weight_file}")

# Create predictions
P = []

s = 0

for X_sim in X_B:

  print(sim_folders[s])
  s+=1

  # Save the first step as it contains the original flags and source
  x_first = X_sim[0]

  x = np.dstack([x_first[:, :, :2], x_first[:,:,1:]])

  P_sim = [x_first]

  for i in range(X_sim.shape[0]):
    x = model.predict(np.array([x]), verbose=False)[0]
    P_sim.append(np.dstack([x_first[:, :, :1], x]))
    # After each prediction extend the result with the original flags and source for the next prediction
    x = np.dstack([x_first[:, :, :2], x])


  P.append(P_sim)

  del P_sim

# Create ste-wise mean simulation
Y_P_M = np.mean(np.asarray(P), axis=(0))
create_files(Y_P_M, Y_B_M, "prediction-01-step-constant")

"""
UNet 10 Steps - Constant
"""

# Load model weights
network_path = f"/content/drive/MyDrive/UNI/Bachelorarbeit/network/UNet_BN_MSE_DET_010_STEPS/NET_000003"

with open(network_path + "/model.json") as f:
  model_info = json.load(f)
  model = tf.keras.models.model_from_json(json.dumps(model_info))

weight_files = sorted(glob.glob(network_path + "/cp*_final.h5"))
weight_file = weight_files[-1]

model.load_weights(weight_file)
print(f"Checkpoint loaded from {weight_file}")

# Create predictions
P = []

s = 0

for X_sim in X_B:

  print(sim_folders[s])
  s+=1

  # Save the first step as it contains the original flags and source
  x_first = X_sim[0]

  P_sim = []

  # Use the first 10 steps as a prediction basis in order to avoid empty spaces throghout the prediction
  P_sim.extend(X_sim[:10])

  x = np.array([np.dstack([x_first[:, :, :2], pred[:,:,1:]]) for pred in P_sim])

  while len(P_sim) < X_sim.shape[0]:
    x = model.predict(x, verbose=False)
    P_sim.extend([np.dstack([x_first[:, :, 0], pred]) for pred in x])
    # Extend each predicted step with the original flags and source for the next prediction
    x = np.array([np.dstack([x_first[:, :, :2], pred]) for pred in x])

  P.append(P_sim[:X_sim.shape[0]+1])

  del P_sim

# Create ste-wise mean simulation
Y_P_M = np.mean(np.asarray(P), axis=(0))
create_files(Y_P_M, Y_B_M, "prediction-10-step-constant")


"""
Prediction 10 steps
"""

# Load model weights
network_path = f"/content/drive/MyDrive/UNI/Bachelorarbeit/network/UNet_BN_MSE_DET_010_STEPS/NET_000000"

with open(network_path + "/model.json") as f:
  model_info = json.load(f)
  model = tf.keras.models.model_from_json(json.dumps(model_info))

weight_files = sorted(glob.glob(network_path + "/cp*_final.h5"))
weight_file = weight_files[-1]

model.load_weights(weight_file)
print(f"Checkpoint loaded from {weight_file}")

# Create predictions
P = []

s = 0

for X_sim in X_B:

  print(sim_folders[s])
  s+=1

  # Use the first 10 steps as a prediction basis in order to avoid empty spaces throghout the prediction
  x = X_sim[:10]

  P_sim = []

  P_sim.extend(x)

  while len(P_sim) < X_sim.shape[0]:
    x = model.predict(x, verbose=False)
    P_sim.extend(x)


  P.append(P_sim[:X_sim.shape[0]+1])

  del P_sim

print(f"Average prediction time for {len(X_sim)} steps: {np.mean(T)}")

# Create ste-wise mean simulation
Y_P_M = np.mean(np.asarray(P), axis=(0))
create_files(Y_P_M, Y_B_M, "prediction-10-step")

"""
Prediciton 20 steps
"""

# Load model weights
network_path = f"/content/drive/MyDrive/UNI/Bachelorarbeit/network/UNet_BN_MSE_DET_020_STEPS/NET_000001"

print(f"Network Path: {network_path}")

with open(network_path + "/model.json") as f:
  model_info = json.load(f)
  model = tf.keras.models.model_from_json(json.dumps(model_info))

weight_files = sorted(glob.glob(network_path + "/cp*_final.h5"))
weight_file = weight_files[-1]

model.load_weights(weight_file)
print(f"Checkpoint loaded from {weight_file}")

# Create predictions
P = []

s = 0

for X_sim in X_B:

  print(sim_folders[s])
  s+=1

  # Use the first 20 steps as a prediction basis in order to avoid empty spaces throghout the prediction
  x = X_sim[:20]

  P_sim = []

  P_sim.extend(x)

  while len(P_sim) < X_sim.shape[0]:
    x = model.predict(x, verbose=False)
    P_sim.extend(x)

  P.append(P_sim[:X_sim.shape[0]+1])

  del P_sim

print(f"Average prediction time for {len(X_sim)} steps: {np.mean(T)}")

# Create ste-wise mean simulation
Y_P_M = np.mean(np.asarray(P), axis=(0))
create_files(Y_P_M, Y_B_M, "prediction-20-step")