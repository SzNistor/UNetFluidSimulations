"""
create_signal_plots.py

Generate plots of the power and magnitude spectra.

Author: Szilard Nistor
"""

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

from copy import copy
from matplotlib import pyplot as plt

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
Get Prediction Path
"""

results_folder = "results_det"

prediction_path = f"/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/RES"
prediction_path = get_unique_path(prediction_path)

make_dir(prediction_path)

plt.rcParams.update({'font.size': 22})

"""
Power Spectrum Plots
"""

for folder in [
    "RELATIVE_ERROR",
    "RESIDUAL",
    "SIGNAL",
]:
  print(folder)
  make_dir(prediction_path +  "/" + folder)
  make_dir(prediction_path +  "/" + folder + "/LOGLOG")

  # Define signals to compare
  simulations = {
      "Vorticity 0.1 - Baseline" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-0_1-power.pickle.gz',
      "Vorticity 0.101" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-0_101-power.pickle.gz',
      "Vorticity 0.15" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-0_15-power.pickle.gz',
      "Vorticity 0.3" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-0_3-power.pickle.gz',
      "Vorticity 0.5" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-0_5-power.pickle.gz',
      "Vorticity 1" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-1-power.pickle.gz',
      "UNet" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/results/{folder}/prediction-01-step-power.pickle.gz',
      "UNet Det" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/prediction-01-step-power.pickle.gz',
      
      "UNet Det 10 Steps" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/prediction-10-step-power.pickle.gz',
      "UNet Det 20 Steps" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/prediction-20-step-power.pickle.gz',

      "UNet Det Const" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/prediction-01-step-constant-power.pickle.gz',
      "UNet Det Const 10 Steps " : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/prediction-10-step-constant-power.pickle.gz',
  }

  simulations_dict = {}

  # Load the power spectrum dictionary for each signal
  for sim, sim_file in simulations.items():
    with gzip.open(sim_file, 'rb') as dict_file:
      sim_dict = pickle.load(dict_file)
      simulations_dict[sim] = sim_dict
  
  # Generate plots at the desired steps
  for i in [1,2,3,4,5,6,7,8,9,14, 24, 49, 99, 199, 299]:
    fig = plt.figure(figsize=(10,15))
    ax = plt.gca()
    plt.title(f"Step {i+1}")
    lines = []
    for sim_label, sim in simulations_dict.items():
      if (folder == "RELATIVE_ERROR" or folder == "RESIDUAL") and sim_label == "Vorticity 0.1 - Baseline":
        continue
      l, = ax.loglog(list(sim.keys()), [v[i] for _, v in sim.items()], label=sim_label)
      lines.append(l)
    #plt.legend() 
    plt.savefig(f"{prediction_path}/{folder}/LOGLOG/STEP_{i+1}_WAVELENGTH.png",bbox_inches='tight')
    plt.clf()
  
  # Create one legend
  legendFig = plt.figure("Legend plot")
  if folder == "SIGNAL":
    legendFig.legend(lines, list(simulations.keys())[:], loc='center', ncol=len(lines) // 2 )
  else:
    legendFig.legend(lines, list(simulations.keys())[1:], loc='center', ncol=len(lines) // 2 )

  legendFig.savefig(f"{prediction_path}/{folder}/LOGLOG/legend.png", bbox_inches="tight")


"""
2D Magnitude Spectrum Plots
"""

for folder in [
    "RELATIVE_ERROR",
    "RESIDUAL",
    "SIGNAL",
    ]:
  print(folder)
  make_dir(prediction_path +  "/" + folder)
  make_dir(prediction_path +  "/" + folder + "/2D")

  for band in [
      "x0", "x32",
      "y0", "y32"
  ]:
    make_dir(prediction_path +  "/" + folder + "/2D/BAND_" + band)

    # Get the first image and draw a red line representing the frequency band
    with gzip.open(f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/start-band.pickle.gz', 'rb') as dict_file:
      first_band = pickle.load(dict_file)

    
    if band == "y32":
      fig = plt.figure(figsize=(8,8))
      plt.title("Frequency Band y = 0" + "\n")

      plt.imshow(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(first_band)))), cmap="plasma", extent=[-32,32,-32,32])
      plt.axhline(0, 0, 1, color="red", linewidth=5.0)
      plt.savefig(f"{prediction_path}/{folder}/2D/BAND_y32/first.png",bbox_inches='tight')
    
    if band == "y0":
      fig = plt.figure(figsize=(8,8))
      plt.title("Frequency Band y = 32" + "\n")

      plt.imshow(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(first_band)))), cmap="plasma", extent=[-32,32,-32,32])
      plt.axhline(31, 0, 1, color="red", linewidth=5.0)
      plt.savefig(f"{prediction_path}/{folder}/2D/BAND_y0/first.png",bbox_inches='tight')

    if band == "x0":
      fig = plt.figure(figsize=(8,8))
      plt.title("Frequency Band x = -32" + "\n")

      plt.imshow(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(first_band)))), cmap="plasma", extent=[-32,32,-32,32])
      plt.axvline(-31, 0, 1, color="red", linewidth=5.0)
      plt.savefig(f"{prediction_path}/{folder}/2D/BAND_x0/first.png",bbox_inches='tight')

    if band == "x32":
      fig = plt.figure(figsize=(8,8))
      plt.title("Frequency Band x = 0" + "\n")

      plt.imshow(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(first_band)))), cmap="plasma", extent=[-32,32,-32,32])
      plt.axvline(0, 0, 1, color="red", linewidth=5.0)
      plt.savefig(f"{prediction_path}/{folder}/2D/BAND_x32/first.png",bbox_inches='tight')

    # Define the signals to compare
    simulations = {
        "Vorticity 0.1 - Baseline" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-0_1-band-{band}.pickle.gz',
        "Vorticity 0.101" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-0_101-band-{band}.pickle.gz',
        "Vorticity 0.15" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-0_15-band-{band}.pickle.gz',
        "Vorticity 0.3" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-0_3-band-{band}.pickle.gz',
        "Vorticity 0.5" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-0_5-band-{band}.pickle.gz',
        "Vorticity 1" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-1-band-{band}.pickle.gz',
        "UNet" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/prediction-01-step-band-{band}.pickle.gz',
    }

    simulations_dict = {}

    # For each signal read the arrays of the frequency band
    for sim, sim_file in simulations.items():
      with gzip.open(sim_file, 'rb') as dict_file:
        sim_bands = pickle.load(dict_file)
        simulations_dict[sim] = sim_bands

    # Min and Max values for normalisation
    min_v = min([np.min(np.log1p(np.asarray(v))) for _,v in simulations_dict.items()])
    max_v = max([np.max(np.log1p(np.asarray(v))) for _,v in simulations_dict.items()])

    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=min_v, vmax=max_v)

    # Plot the development as an image
    for sim_label, sim in simulations_dict.items():
      fig = plt.figure(figsize=(18,8))
      ax = fig.add_subplot(111)
      plt.title(sim_label + "\n")
      plt.imshow(np.log1p(np.asarray(sim).T), extent=[0, 300, -32,32], cmap=cmap, norm=norm)

      plt.savefig(f"{prediction_path}/{folder}/2D/BAND_{band}/{sim_label}.png",bbox_inches='tight')
      plt.clf()

    fig = plt.figure(figsize=(18,8))

    plt.imshow(np.array([[0,1]]))
    plt.gca().set_visible(False)
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
    plt.savefig(f"{prediction_path}/{folder}/2D/BAND_{band}/colorbar.png",bbox_inches='tight')

    plt.clf()

"""
3D Magnitude Spectrum Plots
"""

for folder in [
    "RELATIVE_ERROR",
    "RESIDUAL",
    "SIGNAL",
]:
  print(folder)
  make_dir(prediction_path +  "/" + folder)
  make_dir(prediction_path +  "/" + folder + "/CIRCLE")
  make_dir(prediction_path +  "/" + folder + "/3D")
  make_dir(prediction_path +  "/" + folder + "/3D_FLAT")

  # Define the signals to compare
  simulations = {
      "Vorticity 0.1 - Baseline" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-0_1-magnitude.pickle.gz',
      "Vorticity 0.101" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-0_101-magnitude.pickle.gz',
      "Vorticity 0.15" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-0_15-magnitude.pickle.gz',
      "Vorticity 0.3" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-0_3-magnitude.pickle.gz',
      "Vorticity 0.5" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-0_5-magnitude.pickle.gz',
      "Vorticity 1" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/vorticity-1-magnitude.pickle.gz',
      "UNet" : f'/content/drive/MyDrive/UNI/Bachelorarbeit/{results_folder}/{folder}/prediction-01-step-magnitude.pickle.gz',
  }

  simulations_dict = {}
  
  # For each signal read the dictionaries of the magnitude spectra
  for sim, sim_file in simulations.items():
    with gzip.open(sim_file, 'rb') as dict_file:
      sim_dict = pickle.load(dict_file)
      simulations_dict[sim] = sim_dict
  
  # Min and Max values for normalisation
  min_v = min([min([min(np.log1p(v)) for _,v in simulations_dict[k].items()]) for k in simulations_dict])
  max_v = max([max([max(np.log1p(v)) for _,v in simulations_dict[k].items()]) for k in simulations_dict])

  cmap = plt.cm.plasma
  norm = plt.Normalize(vmin=min_v, vmax=max_v)

  
  # For each simualtion, at each step, display each magnitude spectra as a circle
  for sim_label, sim in simulations_dict.items():
    print(sim_label)
    fig = plt.figure(figsize=(18,8))
    ax = fig.add_subplot(111, projection="3d")
    plt.title(sim_label)
    for i in reversed(range(0, len(sim[0]))):
      for j, freq in enumerate(sim):
        radius = freq * 64
        # Cut the cricle, so the inside of the cylinder is visible later on
        angle = np.linspace(-np.pi * 0.1, 1.25 * np.pi, 360)

        # For freq=0 increase the radius
        if freq == 0:
          radius = 0.01
          angle = np.linspace(0, 2 * np.pi, 360)

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        # Plot every 20th circle
        if j % 20 == 0:
          plt.plot([i] * len(angle), y, x, color = cmap(norm(np.log1p(sim[freq][i]))), linewidth=4)

    ax.view_init(15, -110)
    plt.savefig(f"{prediction_path}/{folder}/CIRCLE/{sim_label}.png",bbox_inches='tight')
    plt.clf()
  
  # For each simualtion, at each step, display each magnitude spectra as a surface plot
  for sim_label, sim in simulations_dict.items():
    print(sim_label)
    x, y = np.meshgrid([f*64 for f in list(sim.keys())], np.arange(300))

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111,projection='3d')
    plt.title(sim_label)

    img = ax.plot_surface(x, y, np.array([np.log1p(v) for k,v in sim.items()]).T, rstride=1, cstride=1, cmap=cmap, edgecolor='none', norm=norm);
    ax.view_init(20, -80)
    plt.savefig(f"{prediction_path}/{folder}/3D/{sim_label}.png",bbox_inches='tight')
    plt.clf()

  fig = plt.figure(figsize=(18,8))

  # Create one colorbar
  plt.imshow(np.array([[0,1]]))
  plt.gca().set_visible(False)
  plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
  plt.savefig(f"{prediction_path}/{folder}/3D/colorbar.png",bbox_inches='tight')
  plt.savefig(f"{prediction_path}/{folder}/3D_FLAT/colorbar.png",bbox_inches='tight')
  plt.savefig(f"{prediction_path}/{folder}/CIRCLE/colorbar.png",bbox_inches='tight')

  plt.clf()