# SCRIPT TO CONVERT AUDIO TO SPECTROGRAM (.NPY)
# This script loads audio files, converts them into spectrograms (Mel or linear),
# applies optional normalization, and saves them in NumPy format (.npy).


import torchaudio
import torchaudio.transforms as T
import numpy as np
import torch
import os

def save_spectrogram_npy(
    audio_path,
    output_path,
    n_fft=1022,
    hop_length=495,
    to_db=True,
    mel=False,
    sample_rate=22050,
    n_mels=128,
    max_duration_sec=30,
    normalize = True,
    mean=-10.2395,
    std=9.8414
):
    waveform, sr = torchaudio.load(audio_path)

    if sr != sample_rate:
        print(f" Resampling from {sr} Hz to {sample_rate} Hz...")
        resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        print(f" Audio has {waveform.shape[0]} channels. Converting to mono...")
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    '''
    # Limit to 30 seconds
    max_samples = sample_rate * max_duration_sec
    if waveform.shape[1] > max_samples:
        print(f" Trimming audio to {max_duration_sec} seconds...")
        waveform = waveform[:, :max_samples]
    '''    

    if mel:
        transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
    else:
        transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
            center=False
        )

    spec = transform(waveform)

    if to_db:
        spec = T.AmplitudeToDB(stype='power', top_db=80)(spec)

    if normalize:
        spec = (spec - mean) / std

    spec_np = spec.squeeze(0).numpy()
    print(f" Spectrogram shape from '{audio_path}': {spec_np.shape}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, spec_np)
    print(f" Saved at: {output_path}")


mean_clean=-19.4780
std_clean=19.4059
mean_noisy=-10.2395
std_noisy=9.8264

mean_test = -10.2395
std_test = 9.8414


#### AUDIO 10 (clean with stats of test and predicted with stats of train)
save_spectrogram_npy(
    "data/full_tracks/audio00010.wav",
    "data/pruebas_data_3/audio00010.npy",
    mean = mean_test,
    std = std_test
)
save_spectrogram_npy(
    "data/full_tracks_noisy/audio00010.wav",
    "data/pruebas_data_3/audio00010_noisy.npy",
    mean = mean_noisy,
    std = std_noisy
)
save_spectrogram_npy(
    "data/full_tracks_pred/audio00010.wav",
    "data/pruebas_data_3/audio00010_pred.npy",
    mean = mean_clean,
    std = std_clean
)
#### PRED AUDIO 10 (predicted with stats of test)
save_spectrogram_npy(
    "data/full_tracks_pred/audio00010.wav",
    "data/pruebas_data_3/audio00010_pred_test_n.npy",
    #hop_length=480,
    mean = mean_test,
    std = std_test
)


#### PRED AUDIO 11 (clean with stats of test and predicted with stats of train)
save_spectrogram_npy(
    "data/full_tracks/audio00011.wav",
    "data/pruebas_data_3/audio00011.npy",
    #mean = mean_clean,
    #std = std_clean
)
save_spectrogram_npy(
    "data/full_tracks_noisy/audio00011.wav",
    "data/pruebas_data_3/audio00011_noisy.npy",
    mean = mean_noisy,
    std = std_noisy
)
save_spectrogram_npy(
    "data/full_tracks_pred/audio00011.wav",
    "data/pruebas_data_3/audio00011_pred.npy",
    mean = mean_clean,
    std = std_clean
)
#### PRED AUDIO 11 (predicted with stats of test)
save_spectrogram_npy(
    "data/full_tracks_pred/audio00011.wav",
    "data/pruebas_data_3/audio00011_pred_test_n.npy",
    #hop_length=480,
    mean = mean_test,
    std = std_test
)


# Segment 17

# Clean with test stats
save_spectrogram_npy(
    "data/split_data/test/audio00010_segment_17.wav",
    "data/pruebas_data/audio00010_segment_17_test_n.npy",
    mean = mean_test,
    std = std_test
)

# Clean with train stats
save_spectrogram_npy(
    "data/split_data/test/audio00010_segment_17.wav",
    "data/pruebas_data/audio00010_segment_17_train_n.npy",
    mean = mean_clean,
    std = std_clean
)

# Predicted with test stats
save_spectrogram_npy(
    "data/seg_to_join/audio00010_segment_17_recon.wav",
    "data/pruebas_data/audio00010_segment_17_pred_test_n.npy",
    hop_length=480,
    mean = mean_test,
    std = std_test
)
