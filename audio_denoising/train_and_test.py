# VERSION FOR NO PADDING, NFFT=1022, HOP_LENGTH=495
# IMPORTANT: This script is intended only for the following noise classes:
# AddWhiteNoise, AddQuantizationNoise, AddQuantizationNoiseWithDither
# It is NOT intended for AddRecordedNoise

#from unet import UNet
from monai_unet import UNet
from dataset import SpectrogramDataset
from AddWhiteNoise import AddWhiteNoise
from AddQuantizationNoise import AddQuantizationNoise
from AddQuantizationNoiseWithDither import AddQuantizationNoiseWithDither
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

TARGET_SHAPE = (512, 88)

mean_noisy=-10.2395
std_noisy=9.8264
mean_clean=-19.4780
std_clean=19.4059

def train_epoch(model, dataloader, criterion, optimizer, device):
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
        '''
        # Monitoring every 50 batches
        if i % 50 == 0:
            print(f"[Batch {i}] GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"[Batch {i}] GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            print(f"[Batch {i}] GPU max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        '''
    return running_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device, return_predictions= False, return_filenames = False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_filenames = []
    all_noisy = []
    psnr_values = []
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
                print("The model produced NaN or inf in predictions")
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
    _, _, h, w = tensor.shape
    pad_h = target_shape[0] - h
    pad_w = target_shape[1] - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)

def compute_psnr(pred, target, mean, std, max_val=80.0, eps=1e-8):
    # Denormalize
    pred_denorm = pred * std + mean
    target_denorm = target * std + mean
    
    mse = F.mse_loss(pred_denorm, target_denorm, reduction='mean')
    psnr = 10 * torch.log10(max_val ** 2 / (mse + eps))
    return psnr.item()

import csv

def save_results_to_csv(filename, data, header=None):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists and header:
            writer.writerow(header)
        writer.writerow(data)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # Files
    train_files = glob("data/split_data/train/*.wav")
    val_files   = glob("data/split_data/val/*.wav")
    test_files  = glob("data/split_data/test/*.wav")

    # Train tracks stats:
    # Mean: -0.000062
    # Std: 0.118354
    # Min: -0.996384
    # Max: 0.997208

    # Std values and their proportion to signals std:
    # std_noise = 0.0012 -> 1%
    # std_noise = 0.006 -> 5%
    # std_noise = 0.012 -> 10%
    # std_noise = 0.024 -> 20%
    # std_noise = 0.035 -> 30%
    # std_noise = 0.06 -> 50%
    
    # Transformations
    train_noise = AddQuantizationNoiseWithDither(num_bits = 6, dither_type='uniform')
    val_test_noise = AddQuantizationNoiseWithDither(num_bits = 6, dither_type='uniform')

    # Datasets
    train_dataset = SpectrogramDataset(train_files, noise_transform=train_noise, mode='train')
    val_dataset   = SpectrogramDataset(val_files, noise_transform=val_test_noise, mode='val')
    test_dataset  = SpectrogramDataset(test_files, noise_transform=val_test_noise, mode='test')

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=16, num_workers=4, pin_memory=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)

    # Loss and optimizer
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    val_losses = []
    val_r2s = []
    best_val_loss = float("inf")
    best_model_state = None
    num_epochs = 30
    patience = 8
    epochs_without_improvement = 0

    import time
    start = time.time()
    for batch in train_loader:
        break
    print(f"Time to load 1 batch from train_loader: {time.time() - start:.3f} seconds")

    if isinstance(train_noise, AddWhiteNoise):
        noise_param = train_noise.std
        header = ["Loss function", "Noise (std)", "Train Loss", "Val Loss", "Val R^2", "Val PSNR", "Epoch", "Test Loss", "Test R^2", "Test PSNR"]
    elif isinstance(train_noise, AddQuantizationNoise):
        noise_param = train_noise.num_bits
        header = ["Loss function", "Nbits", "Train Loss", "Val Loss", "Val R^2", "Val PSNR", "Epoch", "Test Loss", "Test R^2", "Test PSNR"]
    elif isinstance(train_noise, AddQuantizationNoiseWithDither):
        noise_param = f"{train_noise.num_bits} bits, {train_noise.dither_type} dither"
        header = ["Loss function", "Quantization + Dither", "Train Loss", "Val Loss", "Val R^2", "Val PSNR", "Epoch", "Test Loss", "Test R^2", "Test PSNR"]
    else:
        noise_param = None

    csv_filename = "training_results.csv"
    training_start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        train_losses.append(train_loss)

        val_loss, val_r2, val_psnr = validate_epoch(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f} - R2: {val_r2:.4f} - PSNR: {val_psnr:.2f} dB")
        val_losses.append(val_loss)
        val_r2s.append(val_r2)

        test_loss = None
        test_r2 = None
        test_psnr = None

        save_results_to_csv(csv_filename,
                            [type(criterion).__name__, noise_param, train_loss, val_loss, val_r2, val_psnr, epoch + 1, test_loss, test_r2, test_psnr],
                            header=header if epoch == 0 else None)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }
            print("New best model saved!")
            epochs_without_improvement = 0

            os.makedirs("checkpoints", exist_ok=True)
            torch.save(best_model_state, 'checkpoints/best_model_full.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

    # Save final best model
    torch.save(best_model_state['model_state_dict'], "best_model.pth")
    print(f"\nBest validation loss: {best_val_loss:.4f}")

    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    print(f"\nTotal training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")

    # Test evaluation
    model.load_state_dict(torch.load("best_model.pth"))

    test_loss, test_r2, test_psnr, test_preds, test_noisy, test_filenames = validate_epoch(
        model, test_loader, criterion, device, return_predictions=True, return_filenames=True
    )
    print(f"\nFinal Test Loss: {test_loss:.4f} - Test R2: {test_r2:.4f} - Test PSNR: {test_psnr:.2f} dB")

    if isinstance(val_test_noise, AddWhiteNoise):
        noise_param_test = val_test_noise.std
    elif isinstance(val_test_noise, AddQuantizationNoise):
        noise_param_test = val_test_noise.num_bits
    else:
        noise_param_test = None

    save_results_to_csv(csv_filename,
                        [type(criterion).__name__, noise_param_test, None, None, None, None, "Final", test_loss, test_r2, test_psnr])

    os.makedirs("data/predicted_data", exist_ok=True)
    os.makedirs("data/noisy_test_data", exist_ok=True)

    for pred, noisy, fname in zip(test_preds, test_noisy, test_filenames):
        base_name = os.path.splitext(os.path.basename(fname))[0]

        if not np.isfinite(pred[0]).all():
            print(f" Pred[0] contains NaN or inf: {base_name}")
            continue
        if not np.isfinite(noisy[0]).all():
            print(f" Noisy[0] contains NaN or inf: {base_name}")
            continue

        pred_tensor = torch.tensor(pred[0])
        if pred_tensor.ndim == 3 and pred_tensor.shape[0] == 1:
            pred_tensor = pred_tensor.squeeze(0)

        pred_recon = pred_tensor
        pred_recon_np = pred_recon.cpu().numpy()

        noisy_tensor = torch.tensor(noisy[0])
        if noisy_tensor.ndim == 3 and noisy_tensor.shape[0] == 1:
            noisy_tensor = noisy_tensor.squeeze(0)

        noisy_recon = noisy_tensor
        noisy_recon_np = noisy_recon.cpu().numpy()

        pred_path = os.path.join("data/predicted_data", f"{base_name}_pred.npy")
        noisy_path = os.path.join("data/noisy_test_data", f"{base_name}_noisy.npy")

        try:
            np.save(pred_path, pred_recon_np)
            np.save(noisy_path, noisy_recon_np)
        except Exception as e:
            print(f" Error saving {base_name}: {e}")

    # Plot losses and R2
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(val_r2s, label="Validation R2", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("R2 Score")
    plt.title("Validation R2 Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("loss_and_r2_plot.png")
    plt.show()
