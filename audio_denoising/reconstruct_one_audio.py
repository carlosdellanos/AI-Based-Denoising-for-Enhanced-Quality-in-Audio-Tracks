# SCRIPT TO RECONSTRUCT AUDIOS FROM SPECTROGRAMS
# This script reconstructs WAV files from dB spectrograms using Griffin-Lim.
# You can filter to process only specific files using `filter_str`.

import torch
import torchaudio
import numpy as np
import os
from glob import glob

def reconstruct_audios_from_spectrograms(
    input_folder,
    output_folder,
    suffix_in=".npy",
    suffix_out=".wav",
    replace_str=None,
    n_fft=1022,
    hop_length=495,
    sample_rate=22050,
    n_iter=80,
    mean=-10.2395,
    std=9.8414,
    filter_str=None  # NEW: only process files containing this substring
):
    """
    Reconstruct audio from dB spectrograms using Griffin-Lim.

    Args:
        input_folder (str): Folder containing .npy spectrograms
        output_folder (str): Folder to save reconstructed .wav files
        suffix_in (str): Input file suffix (e.g., "_pred.npy" or "_noisy.npy")
        suffix_out (str): Output file suffix
        replace_str (str): Text to replace in the filename
        n_fft (int): FFT size
        hop_length (int): Hop length
        sample_rate (int): Audio sample rate
        n_iter (int): Griffin-Lim iterations
        mean (float): Mean used for denormalization
        std (float): Standard deviation used for denormalization
        filter_str (str): Only process files containing this substring
    """
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
        n_iter=n_iter
    )

    def db_to_amplitude(db_tensor):
        # Convert from dB to linear power
        return torch.pow(10.0, db_tensor / 10.0)

    os.makedirs(output_folder, exist_ok=True)

    for file_path in glob(os.path.join(input_folder, f"*{suffix_in}")):
        base_name = os.path.basename(file_path)

        # Skip files not matching filter
        if filter_str is not None and filter_str not in base_name:
            continue

        spec_db = np.load(file_path)
        spec_tensor = torch.tensor(spec_db, dtype=torch.float32)

        # --- Denormalization ---
        spec_db = spec_tensor * std + mean
        spec_mag = db_to_amplitude(spec_db)

        waveform = griffin_lim(spec_mag)

        # Determine output filename
        if replace_str:
            file_name = base_name.replace(replace_str, suffix_out)
        else:
            file_name = base_name.replace(suffix_in, suffix_out)

        save_path = os.path.join(output_folder, file_name)
        torchaudio.save(save_path, waveform.unsqueeze(0), sample_rate=sample_rate)
        #print(f"Saved: {save_path}")

# Dataset statistics
mean_noisy = -10.2395
std_noisy = 9.8264
mean_clean = -19.4780
std_clean = 19.4059

# Reconstruct only files containing "audio00011"
reconstruct_audios_from_spectrograms(
    input_folder="data/predicted_data",
    output_folder="data/seg_to_join",
    replace_str="_pred.npy",
    suffix_out="_recon.wav",
    mean=mean_clean,
    std=std_clean,
    filter_str="audio00011"
)

reconstruct_audios_from_spectrograms(
    input_folder="data/noisy_test_data",
    output_folder="data/seg_to_join_noisy",
    replace_str="_noisy.npy",
    suffix_out="_noisy_recon.wav",
    mean=mean_noisy,
    std=std_noisy,
    filter_str="audio00011"
)
