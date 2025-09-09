# VERSION FOR TESTING AND SAVING PREDICTIONS
# IMPORTANT: This script is intended only for the following noise classes:
# AddWhiteNoise, AddQuantizationNoise, AddQuantizationNoiseWithDither
# It is NOT intended for AddRecordedNoise

import os
import torch
from monai_unet import UNet
from dataset import SpectrogramDataset
from AddWhiteNoise import AddWhiteNoise
from AddQuantizationNoise import AddQuantizationNoise
from AddQuantizationNoiseWithDither import AddQuantizationNoiseWithDither
from AddRecordedNoise import AddRecordedNoise
from torch.utils.data import DataLoader
from PSNRLoss import PSNRLoss
from sklearn.metrics import r2_score
import numpy as np
from glob import glob

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test files
test_files = glob("data/split_data/test/*.wav")

# Transform for test (no new noise added)
# val_test_noise = AddWhiteNoise(std_noise=0.06, train=False)
val_test_noise = AddQuantizationNoiseWithDither(num_bits=6, dither_type='uniform')
# val_test_noise = AddQuantizationNoise(num_bits=10)
test_dataset = SpectrogramDataset(test_files, noise_transform=val_test_noise, mode='test')
test_loader = DataLoader(test_dataset, batch_size=16)

# Load model (same as during training)
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
).to(device)

# Loss criterion
# criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss(beta=1.0)
# criterion = PSNRLoss(max_val=3.0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Load saved weights
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# --- Dataset statistics (same as used during training) ---
mean_noisy = -10.2395
std_noisy  = 9.8264
mean_clean = -19.4780
std_clean  = 19.4059

def compute_psnr(pred, target, mean, std, max_val=80.0, eps=1e-8):
    # Denormalize
    pred_denorm = pred * std + mean
    target_denorm = target * std + mean

    mse = F.mse_loss(pred_denorm, target_denorm, reduction='mean')
    psnr = 10 * torch.log10(max_val ** 2 / (mse + eps))
    return psnr.item()

def test_model_and_save_preds(model, dataloader, criterion, device):
    running_loss = 0
    all_preds = []
    all_targets = []
    all_noisy = []
    psnr_values = []
    filenames = []

    os.makedirs("data/predicted_data", exist_ok=True)
    os.makedirs("data/noisy_test_data", exist_ok=True)

    with torch.no_grad():
        for noisy_spctr, clean_spctr, file_names in dataloader:
            noisy_spctr, clean_spctr = noisy_spctr.to(device), clean_spctr.to(device)
            outputs = model(noisy_spctr)
            loss = criterion(outputs, clean_spctr)
            running_loss += loss.item()

            # PSNR (denormalized)
            batch_psnr = compute_psnr(outputs, clean_spctr, mean_clean, std_clean)
            psnr_values.append(batch_psnr)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(clean_spctr.cpu().numpy())
            all_noisy.append(noisy_spctr.cpu().numpy())
            filenames.extend(file_names)

    avg_loss = running_loss / len(dataloader)
    avg_psnr = np.mean(psnr_values)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_noisy = np.concatenate(all_noisy, axis=0)

    r2 = r2_score(all_targets.reshape(-1), all_preds.reshape(-1))

    # Save predictions and noisy spectrograms
    for pred, noisy, fname in zip(all_preds, all_noisy, filenames):
        base_name = os.path.splitext(os.path.basename(fname))[0]

        # Validate NaNs or infs
        if not np.isfinite(pred[0]).all():
            print(f"Prediction contains NaN or inf: {base_name}")
            continue
        if not np.isfinite(noisy[0]).all():
            print(f"Noisy contains NaN or inf: {base_name}")
            continue

        pred_tensor = torch.tensor(pred[0])
        if pred_tensor.ndim == 3 and pred_tensor.shape[0] == 1:
            pred_tensor = pred_tensor.squeeze(0)
        pred_recon_np = pred_tensor.cpu().numpy()

        noisy_tensor = torch.tensor(noisy[0])
        if noisy_tensor.ndim == 3 and noisy_tensor.shape[0] == 1:
            noisy_tensor = noisy_tensor.squeeze(0)
        noisy_recon_np = noisy_tensor.cpu().numpy()

        pred_path = os.path.join("data/predicted_data", f"{base_name}_pred.npy")
        noisy_path = os.path.join("data/noisy_test_data", f"{base_name}_noisy.npy")

        try:
            np.save(pred_path, pred_recon_np)
            np.save(noisy_path, noisy_recon_np)
        except Exception as e:
            print(f"Error saving {base_name}: {e}")

    return avg_loss, r2, avg_psnr

if __name__ == "__main__":
    test_loss, test_r2, test_psnr = test_model_and_save_preds(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} - Test R2: {test_r2:.4f} - Test PSNR: {test_psnr:.2f} dB")
