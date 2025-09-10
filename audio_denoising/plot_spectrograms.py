# SCRIPT TO PLOT SPECTROGRAM COMPARISONS
# This script plots clean, noisy, and reconstructed spectrograms side by side.
# Each plot is saved in SVG format inside the "figures" folder.

import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure that the figures folder exists
os.makedirs("figures", exist_ok=True)

def plot_spectrograms(clean_path, noisy_path, recon_path, title=None, filename=None):
    """
    Plot clean, noisy, and reconstructed spectrograms, then save them as SVG.
    """

    # Load spectrograms
    clean = np.load(clean_path)
    noisy = np.load(noisy_path)
    recon = np.load(recon_path)

    # Print shapes
    print(f"Shape clean: {clean.shape}")
    print(f"Shape noisy: {noisy.shape}")
    print(f"Shape recon: {recon.shape}")

    # Ensure they match
    assert clean.shape == noisy.shape == recon.shape, "Spectrogram shapes do not match"

    # Create figure
    plt.figure(figsize=(15, 4))

    # Common limits
    vmin = min(clean.min(), noisy.min(), recon.min())
    vmax = max(clean.max(), noisy.max(), recon.max())

    # Clean
    plt.subplot(1, 3, 1)
    plt.imshow(clean, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
    plt.title("Original")
    plt.xlabel("Time (frames)")
    plt.ylabel("Frequency (bins)")
    plt.colorbar()

    # Noisy
    plt.subplot(1, 3, 2)
    plt.imshow(noisy, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
    plt.title("Noisy")
    plt.xlabel("Time (frames)")
    plt.ylabel("Frequency (bins)")
    plt.colorbar()

    # Reconstructed
    plt.subplot(1, 3, 3)
    plt.imshow(recon, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
    plt.title("Denoised")
    plt.xlabel("Time (frames)")
    plt.ylabel("Frequency (bins)")
    plt.colorbar()

    if title:
        plt.suptitle(title)

    plt.tight_layout()

    # Save as SVG
    if filename:
        save_path = os.path.join("figures", f"{filename}.svg")
        plt.savefig(save_path, format="svg")
        print(f"Figure saved at: {save_path}")
        plt.show()
    else:
        plt.show()


# (clean with test stats, pred with train stats)
plot_spectrograms("data/pruebas_data/audio00010_segment_17_test_n.npy", 
                  "data/noisy_test_data/audio00010_segment_17_noisy.npy",
                  "data/predicted_data/audio00010_segment_17_pred.npy",
                  filename="audio_00010_segment_17_test_train")

# (clean with train stats, pred with train stats)
plot_spectrograms("data/pruebas_data/audio00010_segment_17_train_n.npy", 
                  "data/noisy_test_data/audio00010_segment_17_noisy.npy",
                  "data/predicted_data/audio00010_segment_17_pred.npy",
                  filename="audio_00010_segment_17_train_train")

# (clean and pred both with test stats, both from audio not network)
plot_spectrograms("data/pruebas_data/audio00010_segment_17_test_n.npy", 
                  "data/noisy_test_data/audio00010_segment_17_noisy.npy",
                  "data/pruebas_data/audio00010_segment_17_pred_test_n.npy",
                  filename="audio_00010_segment_17_test_test")


# Audio 10 (clean with test stats, pred with train stats)
plot_spectrograms("data/pruebas_data_3/audio00010.npy", 
                  "data/pruebas_data_3/audio00010_noisy.npy",
                  "data/pruebas_data_3/audio00010_pred.npy",
                  filename="audio_00010_test_train")

# Audio 10 (clean with test stats, pred with test stats)
plot_spectrograms("data/pruebas_data_3/audio00010.npy", 
                  "data/pruebas_data_3/audio00010_noisy.npy",
                  "data/pruebas_data_3/audio00010_pred_test_n.npy",
                  filename="audio_00010_test_test")


# Audio 11 (clean with test stats, pred with train stats)
plot_spectrograms("data/pruebas_data_3/audio00011.npy", 
                  "data/pruebas_data_3/audio00011_noisy.npy",
                  "data/pruebas_data_3/audio00011_pred.npy",
                  filename="audio_00011_test_train")

# Audio 11 (clean with test stats, pred with test stats)
plot_spectrograms("data/pruebas_data_3/audio00011.npy", 
                  "data/pruebas_data_3/audio00011_noisy.npy",
                  "data/pruebas_data_3/audio00011_pred_test_n.npy",
                  filename="audio_00011_test_test")
