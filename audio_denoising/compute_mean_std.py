# VERSION TO CALCULATE MEAN AND STD
# IMPORTANT: This script is intended only for the following noise classes:
# AddWhiteNoise, AddQuantizationNoise, AddQuantizationNoiseWithDither
# It is NOT intended for AddRecordedNoise

import torch
from torch.utils.data import DataLoader
from dataset import SpectrogramDataset
from AddWhiteNoise import AddWhiteNoise
from glob import glob
import numpy as np

# List of files
files = glob("data/split_data/test/*.wav")

# Dataset in 'train' mode (to include noise if used during training)
noise_transform = AddWhiteNoise(std_noise=0.01, train=True)
dataset = SpectrogramDataset(files, mode='train', noise_transform=noise_transform)

# DataLoader without shuffling
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialization
all_values = []

print("Calculating mean and standard deviation...")

for noisy_db, _ in loader:
    # noisy_db: (1, 1, H, W)
    noisy_db = noisy_db.squeeze().flatten()  # remove batch and channel, then flatten
    all_values.append(noisy_db.numpy())

# Concatenate all values and compute statistics
all_values = np.concatenate(all_values)
mean = np.mean(all_values)
std = np.std(all_values)

print(f"\nMean: {mean:.4f}")
print(f"Standard deviation: {std:.4f}")
