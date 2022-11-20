"""
dataset_analysis.py

Script for retrieving the means and standard deviations of a datasets.
The script was originally an .ipynb file used in Google Colab.

Author: Szilard Nistor
"""

"""
Basic Imports
"""

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Add folder containing simualtions to path
import sys
sys.path.append("/content/drive/MyDrive/UNI/Bachelorarbeit")

from util.data import *

import glob

import multiprocessing as mp
import numpy as np
import pandas as pd

from IPython.core.display import clear_output
from copy import copy

dataset = "TRAIN_DET"

"""
Get Simulation IDs
"""

def get_sim_scenes(sim):
  files = glob.glob(f"{sim}/combined/*.npz")
  
  sim_ids = [f for f in files if "label" not in f]

  return sim_ids

def save_sim_scenes(result):
  global ids

  ids.extend(result)

ids = []

pool = mp.Pool(32)
results = []

simulations = glob.glob(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/{dataset}/*")

for sim in simulations:
  r = pool.apply_async(get_sim_scenes, (sim,), callback=save_sim_scenes)
  results.append(r)

for i, r in enumerate(results):
  clear_output(wait=True)
  print(f"Gathering Ids: {(i+1)/(len(simulations))*100:.2f} %")
  r.wait()

pool.close()
pool.join()

"""
Get Channel-Wise Mean
"""

def split(arr, chunk_size):
  for i in range(0, len(arr), chunk_size):
    yield arr[i:i + chunk_size]

# Load files in chunks and return the channel-wise sum
def get_sum_mean(chunk_ids):
  sums = np.zeros(6)

  for chunk_id in chunk_ids:
    # Load Id
    x = np.load(chunk_id)["arr_0"]

    # Get Velocity Magnitude
    x_vel = x[:, :, 3:5]
    x = np.dstack([x, np.linalg.norm(x_vel, axis=2)])

    # Remap Flags to 0 and 1
    s = x[:, :, 0]
    # Set empty spaces to 0
    s[s == 1065353216.0] = 0
    # Set borders to 1
    s[s == 1101004800.0 ] = 1
    # Set obstacles and walls to 2
    s[s == 1073741824.0 ] = 2
    x[:, :, 0] = s

    sums += np.sum(x, axis=(0, 1))

    del x

  # Return Channel-Wise Sum
  return sums

def sum_callback_mean(s):
  global dataset_sums_mean
  global ids_count

  dataset_sums_mean += s
  ids_count += 1

id_chunks = list(split(ids, 25))

dataset_sums_mean = np.zeros(6)

ids_count = 0

pool = mp.Pool(8)
results = []

for chunk_ids in id_chunks:
  r = pool.apply_async(get_sum_mean, (chunk_ids,), callback=sum_callback_mean)
  results.append(r)

for i, r in enumerate(results):
  clear_output(wait=True)
  print(f"Calculating sums: {(i/(len(ids) // 25)*100) : .2f} %")
  r.wait()

pool.close()
pool.join()

dataset_means = dataset_sums_mean/(len(ids)*64*64)
pd.DataFrame(dataset_means).to_csv(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/{dataset}/means.csv", index=False)

"""
Get Channel-Wise Standard Deviation
"""

# Load the data in chunks and subtract the means from each channel. Return the error square sum.
def get_sum_std(chunk_ids, means):
  sums = np.zeros(6)

  for chunk_id in chunk_ids:
    # Load Id
    x = np.load(chunk_id)["arr_0"]

    # Get Velocity Magnitude
    x_vel = x[:, :, 3:5]
    x = np.dstack([x, np.linalg.norm(x_vel, axis=2)])

    # Remap Flags to 0 and 1
    s = x[:, :, 0]
    # Set empty spaces to 0
    s[s == 1065353216.0] = 0
    # Set borders to 1
    s[s == 1101004800.0 ] = 1
    # Set obstacles and walls to 2
    s[s == 1073741824.0 ] = 2
    x[:, :, 0] = s

    # Subtract the Channel Mean
    for i in range(x.shape[2]):
      x[:, :, i] -= means[i]

    # Return Channel Wise Squared Difference
    sums += np.sum(np.square(x), axis=(0, 1))

    del x

  return sums

def sum_callback_std(s):
  global dataset_sums_std
  global ids_count
  
  dataset_sums_std += s
  ids_count += 1

id_chunks = list(split(ids, 25))

dataset_sums_std = np.zeros(6)

pool = mp.Pool(8)
results = []

ids_count = 0

for chunk_ids in id_chunks:
  r = pool.apply_async(get_sum_std, (chunk_ids,dataset_means, ), callback=sum_callback_std)
  results.append(r)

for i, r in enumerate(results):
  clear_output(wait=True)
  print(f"Calculating error square sums: {(i/(len(ids) // 25)*100) : .2f} %")
  r.wait()

pool.close()
pool.join()

# determine the standard deviations
dataset_stds = np.sqrt(dataset_sums_std/(len(ids)*64*64))
pd.DataFrame(dataset_stds).to_csv(f"/content/drive/MyDrive/UNI/Bachelorarbeit/datasets/{dataset}/stds.csv", index=False)

