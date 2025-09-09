# UNet training and validation with spectrograms
# IMPORTANT: This script is intended only for the AddRecordedNoise class
# It is not applicable for AddWhiteNoise or AddQuantizationNoise
# NO PADDING VERSION, NFFT=1022, HOP_LENGTH=495

from monai_unet import UNet
from dataset import SpectrogramDataset
from AddWhiteNoise import AddWhiteNoise
from AddQuantizationNoise import AddQuantizationNoise
from AddRecordedNoise import AddRecordedNoise
from glob import glob
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from PSNRLoss import PSNRLoss

# Target spectrogram shape (height, width)
TARGET_SHAPE = (512, 88)

# Normalization stats
mean_noisy = -10.2395
std_noisy = 9.8264
mean_clean = -19.4780
std_clean = 19.4059


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0

    for i, (noisy_spctr, clean_spctr) in enumerate(dataloader, 1):
        noisy_spctr, clean_spctr = noisy_spctr.to(device), clean_spctr.to(device)

        optimizer.zero_grad()
        outputs = model(noisy_spctr)
        loss = criterion(outputs, clean_spctr)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device, return_predictions=False, return_filenames=False):
    """Validate model on given dataloader, compute loss, R², and PSNR."""
    model.eval()
    running_loss = 0.0
    all_preds, all_targets, all_filenames, all_noisy, psnr_values = [], [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                noisy_spctr, clean_spctr, file_names = batch  
                if return_filenames:
                    all_filenames.extend(file_names)
            else:
                noisy_spctr, clean_spctr = batch

            noisy_spctr, clean_spctr = noisy_spctr.to(device), clean_spctr.to(device)

            outputs = model(noisy_spctr)
            if not torch.isfinite(outputs).all():
                print("Model produced NaN or Inf values in predictions")
                print(f"Noisy input shape: {noisy_spctr.shape}")
                print(f"Clean target shape: {clean_spctr.shape}")

            loss = criterion(outputs, clean_spctr)
            running_loss += loss.item()

            batch_psnr = compute_psnr(outputs, clean_spctr, mean_clean, std_clean)
            psnr_values.append(batch_psnr)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(clean_spctr.cpu().numpy())
            all_noisy.append(noisy_spctr.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_noisy = np.concatenate(all_noisy, axis=0)

    r2 = r2_score(all_targets.reshape(-1), all_preds.reshape(-1))
    avg_psnr = np.mean(psnr_values)

    if return_filenames and return_predictions:
        return avg_loss, r2, avg_psnr, all_preds, all_noisy, all_filenames
    else:
        return avg_loss, r2, avg_psnr


def adjust_to_reconstruct(padded_spec, original_time=87, padded_time=88):
    """Remove padding and restore spectrogram dimensions to original size."""
    total_pad = padded_time - original_time
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left

    if total_pad <= 0:
        cropped = padded_spec
    else:
        cropped = padded_spec[:, pad_left : padded_spec.shape[1] - pad_right]

    if cropped.shape[0] == 512:
        restored = F.pad(cropped, (0, 0, 0, 1))
    else:
        restored = cropped

    return restored


def pad_spectrogram(tensor, target_shape=TARGET_SHAPE):
    """Pad spectrogram tensor to match target shape (center padding)."""
    _, _, h, w = tensor.shape
    pad_h = target_shape[0] - h
    pad_w = target_shape[1] - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)


def compute_psnr(pred, target, mean, std, max_val=80.0, eps=1e-8):
    """Compute PSNR between predicted and target spectrograms (after denormalization)."""
    pred_denorm = pred * std + mean
    target_denorm = target * std + mean
    
    mse = F.mse_loss(pred_denorm, target_denorm, reduction='mean')
    psnr = 10 * torch.log10(max_val ** 2 / (mse + eps))
    return psnr.item()


def extract_song_id(file_path):
    """
    Extract unique song ID from filename of the form:
        audio123_segment_045.wav  -> 1230000 + 45
    Also handles adjusted files:
        audio123_segment_045_adjusted.wav
    """
    file_name = os.path.basename(file_path)
    base = os.path.splitext(file_name)[0]
    track_str, segment_str = base.replace("audio", "").split("_segment_")
    segment_str = segment_str.split("_")[0]
    return int(track_str) * 10000 + int(segment_str)


import csv
def save_results_to_csv(filename, data, header=None):
    """Append training/validation/test results to CSV file (with optional header)."""
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists and header:
            writer.writerow(header)
        writer.writerow(data)


# ----------------------------------------------------------
# Main execution (training, validation, and testing loop)
# ----------------------------------------------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # File paths
    train_files = glob("data/split_data/train/*.wav")
    val_files   = glob("data/split_data/val/*.wav")
    test_files  = glob("data/split_data/test/*.wav")
    recorded_noises_path = "data/real_noise_data/"

    # Example noise transforms (uncomment as needed)
    # train_noise = AddWhiteNoise(std_noise=0.006, train=True)
    # val_test_noise = AddWhiteNoise(std_noise=0.006, train=False)
    # train_noise = AddQuantizationNoise(num_bits=10)
    # val_test_noise = AddQuantizationNoise(num_bits=10)

    SEED = 1234
    train_noise = AddRecordedNoise(recorded_noises_path, mode='train')
    val_noise = AddRecordedNoise(recorded_noises_path, mode='val', seed=SEED)
    test_noise = AddRecordedNoise(recorded_noises_path, mode='test', seed=SEED)

    # Prepare datasets and dataloaders
    train_dataset = SpectrogramDataset(train_files, noise_transform=train_noise, mode='train')
    val_dataset   = SpectrogramDataset(val_files, noise_transform=val_noise, mode='val')
    test_dataset  = SpectrogramDataset(test_files, noise_transform=test_noise, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=16, num_workers=4, pin_memory=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model (UNet from MONAI)
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    # Loss function and optimizer
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses, val_r2s = [], [], []
    best_val_loss = float("inf")
    best_model_state = None
    num_epochs = 30
    patience = 8
    epochs_without_improvement = 0

    # CSV header (noise_param only logged for AddWhiteNoise, AddQuantizationNoise, AddQuantizationNoiseWithDither)
    header = ["Loss function", "Noise (std or bits)", "Train Loss", "Val Loss", "Val R^2", "Val PSNR", "Epoch", "Test Loss", "Test R^2", "Test PSNR"]
    if isinstance(train_noise, AddWhiteNoise):
        noise_param = train_noise.std
    elif isinstance(train_noise, AddQuantizationNoise):
        noise_param = train_noise.num_bits
    else:
        noise_param = None

    csv_filename = "training_results.csv"

    # (Training loop continues as in your code...)
