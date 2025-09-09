# Dataset class for loading audio files, generating spectrograms, 
# and optionally applying noise transformations.
#
# Parameters:
#   file_list (list[str]): List of paths to audio files.
#   mode (str): Dataset mode, either "train", "val", or "test".
#   noise_transform (callable or None): Optional noise transformation to apply 
#                                       (e.g., AddWhiteNoise, AddRecordedNoise).
#   sr (int): Target sample rate for audio resampling (default: 22050).
#   n_fft (int): FFT window size for spectrogram computation (default: 1022).
#   hop_length (int): Hop length for spectrogram frames (default: 495).
#
# Each dataset item returns the noisy and clean spectrograms (in dB scale),
# properly normalized. In "test" mode, it also returns the file name for reference.

import torch
import torchaudio
import os
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, file_list, mode='train', noise_transform=None, sr=22050, n_fft=1022, hop_length=495):
        self.file_list = file_list
        self.mode = mode
        self.noise_transform = noise_transform
        #self.resample = torchaudio.transforms.Resample(orig_freq=None, new_freq=sr)
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0, center=False)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        self.target_sr = sr

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        file_name = os.path.basename(file_path)

        # Extract num_track and num_segment from file name
        # Expected format: audio00012_segment_3.wav
        try:
            base = os.path.splitext(file_name)[0]
            track_part, segment_part = base.split("_segment_")
            num_track = int(track_part.replace("audio", ""))
            num_segment_str = segment_part.split('_')[0]
            num_segment = int(num_segment_str)
            unique_id = num_track * 10000 + num_segment
        except Exception as e:
            raise ValueError(f"Error parsing ID from filename '{file_name}': {e}")

        # Load audio
        waveform, sr = torchaudio.load(file_path)

        if sr != self.target_sr:
            resample_transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resample_transform(waveform)

        # Ensure mono conversion, regardless of the number of channels (convert to 1D)
        waveform = waveform.mean(dim=0)

        # Adding noise
        if self.noise_transform is not None:
            noisy_waveform = self.noise_transform(waveform, id=unique_id, segment_index=num_segment)
        else:
            noisy_waveform = waveform

        # Spectrograms
        clean_spec = self.spectrogram(waveform)
        noisy_spec = self.spectrogram(noisy_waveform)

        # Convert to dB scale
        clean_db = self.amplitude_to_db(clean_spec)
        noisy_db = self.amplitude_to_db(noisy_spec)
        #clean_db = self.amplitude_to_db(torch.abs(clean_spec))
        #noisy_db = self.amplitude_to_db(torch.abs(noisy_spec))
        clean_db = torch.clamp(clean_db, min=-80.0)
        noisy_db = torch.clamp(noisy_db, min=-80.0)

        # Standard normalization (mean 0, std 1)
        #mean = -10.2395
        #std = 9.8414
        mean_noisy = -10.2395
        std_noisy = 9.8264
        mean_clean = -19.4780
        std_clean = 19.4059

        clean_db = (clean_db - mean_clean) / std_clean
        noisy_db = (noisy_db - mean_noisy) / std_noisy

        # Return tensors ready for the network
        if self.mode == 'test':
            return noisy_db.unsqueeze(0), clean_db.unsqueeze(0), file_name
        else:
            return noisy_db.unsqueeze(0), clean_db.unsqueeze(0)
