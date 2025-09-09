import torch
import torchaudio
import numpy as np
import os
from glob import glob

# Function to reconstruct audio signals from spectrograms in dB scale using Griffin-Lim algorithm.
#
# Parameters:
#   input_folder (str): Path to the folder containing .npy spectrogram files.
#   output_folder (str): Path to the folder where reconstructed .wav files will be saved.
#   suffix_in (str): Input file suffix (e.g., "_pred.npy" or "_noisy.npy").
#   suffix_out (str): Output file suffix for the reconstructed audio (e.g., "_recon.wav").
#   replace_str (str): Substring to replace in filenames when saving outputs.
#   n_fft (int): FFT size used to generate the spectrogram.
#   hop_length (int): Hop length used in STFT.
#   sample_rate (int): Target audio sample rate.
#   n_iter (int): Number of Griffin-Lim iterations for phase reconstruction.
#   mean (float): Mean used during spectrogram normalization (for denormalization).
#   std (float): Standard deviation used during spectrogram normalization (for denormalization).
#
# Notes:
#   - The spectrograms are expected to be in dB scale and standardized.
#   - Griffin-Lim is used to estimate the phase and reconstruct the waveform.
#   - Output audio files are saved as .wav with the specified sample rate.

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
    std=9.8414
):
    """Reconstruct audios from spectrograms in dB scale using Griffin-Lim."""
    
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
        n_iter=n_iter
    )

    def db_to_amplitude(db_tensor: torch.Tensor) -> torch.Tensor:
        """Convert dB scale spectrogram to amplitude (power)."""
        return torch.pow(10.0, db_tensor / 10.0)

    os.makedirs(output_folder, exist_ok=True)

    for file_path in glob(os.path.join(input_folder, f"*{suffix_in}")):
        spec_db = np.load(file_path)
        spec_tensor = torch.tensor(spec_db, dtype=torch.float32)

        # --- Denormalization ---
        spec_db = spec_tensor * std + mean

        # Convert from dB to magnitude
        spec_mag = db_to_amplitude(spec_db)

        # Reconstruct waveform using Griffin-Lim
        waveform = griffin_lim(spec_mag)

        # Build output filename
        base_name = os.path.basename(file_path)
        if replace_str:
            file_name = base_name.replace(replace_str, suffix_out)
        else:
            file_name = base_name.replace(suffix_in, suffix_out)

        save_path = os.path.join(output_folder, file_name)
        torchaudio.save(save_path, waveform.unsqueeze(0), sample_rate=sample_rate)


# Normalization stats (used for denormalization)
mean_noisy = -10.2395
std_noisy = 9.8264
mean_clean = -19.4780
std_clean = 19.4059

# Reconstruct predictions
reconstruct_audios_from_spectrograms(
    input_folder="data/predicted_data",
    output_folder="data/reconstructed_data",
    replace_str="_pred.npy",
    suffix_out="_recon.wav",
    mean=mean_clean,
    std=std_clean
)

# Reconstruct noisy audios
reconstruct_audios_from_spectrograms(
    input_folder="data/noisy_test_data",
    output_folder="data/reconstructed_noisy",
    replace_str="_noisy.npy",
    suffix_out="_noisy_recon.wav",
    mean=mean_noisy,
    std=std_noisy
)
