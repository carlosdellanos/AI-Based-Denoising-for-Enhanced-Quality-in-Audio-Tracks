# Class used for adding real recorded noise signals to the original clean signal
#
# Parameters:
#   noise_dir (str): Directory containing .wav noise files
#   psnr_db (float): Desired Peak Signal-to-Noise Ratio in decibels for the signals (or segments of signals) that are loaded in batches 
#   sample_rate (int): Target sample rate for both clean and noise signals (default: 22050)
#   mode (str): "train" for random noise selection, "val"/"test" for deterministic assignment
#   seed (int or None): Random seed for reproducibility (default: None)
#
# When training, for a determined clean signal (segment), the class randomly selects a 
# recorded noise signal from the noise directory and adds a fragment of the same length.  
# During validation/testing, it assigns noise deterministically to ensure reproducibility.

import torchaudio
import torch
import os
import random

class AddRecordedNoise:
    def __init__(self, noise_dir, psnr_db=20, sample_rate=22050, mode='train', seed=None):
        self.noise_paths = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith(".wav")]
        assert len(self.noise_paths) > 0, "No .wav files found in the noise directory"
        self.psnr_db = psnr_db
        self.sample_rate = sample_rate
        self.mode = mode
        self.seed = seed
        self.noise_map = {}    # For deterministic evaluation
        self.used_noises = {}  # Track which noise file was used

    def set_mode(self, mode):
        self.mode = mode

    def assign_noise_map(self, song_ids):
        self.noise_map = {}
        unique_tracks = sorted(set(song_id // 10000 for song_id in song_ids))

        if self.mode == 'train':
            for track_id in unique_tracks:
                noise_path = random.choice(self.noise_paths)
                self.noise_map[track_id] = noise_path
        else:
            sorted_noise_paths = sorted(self.noise_paths)
            for i, track_id in enumerate(unique_tracks):
                noise_path = sorted_noise_paths[i % len(sorted_noise_paths)]
                self.noise_map[track_id] = noise_path

    def _load_noise(self, path, target_len):
        noise, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            noise = torchaudio.functional.resample(noise, sr, self.sample_rate)
        if noise.shape[0] > 1:
            noise = noise.mean(dim=0, keepdim=True)
        else:
            noise = noise[:1]
        noise = noise.squeeze(0)
        if len(noise) < target_len:
            repeats = (target_len // len(noise)) + 1
            noise = noise.repeat(repeats)[:target_len]
        else:
            noise = noise[:target_len]
        return noise

    def _adjust_psnr(self, signal, noise):
        mse_target = (signal.abs().max() ** 2) / (10 ** (self.psnr_db / 10))
        mse_current = torch.mean(noise ** 2)
        scaling_factor = torch.sqrt(mse_target / (mse_current + 1e-8))
        return noise * scaling_factor

    def _normalize_noise(self, noise):
        rms = torch.sqrt(noise.pow(2).mean())
        if rms > 0:
            noise = noise / rms
        return noise

    def __call__(self, x, id=None, segment_index=None):
        x = x.squeeze(0) if x.ndim == 2 else x
        target_len = x.shape[0]

        if id is None:
            raise ValueError("An id is required to assign noise")
        
        num_track = id // 10000

        if num_track not in self.noise_map:
            raise ValueError(f"id {id} -> num_track {num_track} not found in noise_map")

        noise_path = self.noise_map[num_track]

        # print(f"[AddRecordedNoise] Applying noise {os.path.basename(noise_path)} to id {id} (track {num_track})")

        # Register used noise
        self.used_noises[id] = os.path.basename(noise_path)

        # Load full noise signal
        full_noise, sr = torchaudio.load(noise_path)
        if sr != self.sample_rate:
            full_noise = torchaudio.functional.resample(full_noise, sr, self.sample_rate)
        full_noise = full_noise.mean(dim=0) if full_noise.shape[0] > 1 else full_noise[0]

        if self.mode == 'train':
            max_start = max(0, full_noise.shape[0] - target_len)
            start = random.randint(0, max_start)
        else:
            if segment_index is None:
                raise ValueError("segment_index is required in test/val mode for deterministic behavior")
            start = segment_index * target_len

        total_needed_length = start + target_len
        if full_noise.shape[0] < total_needed_length:
            repeats = (total_needed_length // full_noise.shape[0]) + 1
            full_noise = full_noise.repeat(repeats)

        noise = full_noise[start: start + target_len]
        noise = self._normalize_noise(noise)
        noise = self._adjust_psnr(x, noise)
        x_noisy = x + noise

        return x_noisy

    def get_used_noises(self):
        return self.used_noises
