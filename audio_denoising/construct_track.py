# SCRIPT TO JOIN AUDIO SEGMENTS INTO FULL TRACKS
# This script concatenates segmented WAV files into full tracks.
# It can process predicted segments, noisy segments, and original segments.

import os
import torchaudio
import torch
from glob import glob
from collections import defaultdict

# Directories with segmented audio
og_dir = "data/seg_to_join_og_2"      # Original segments
pred_dir = "data/seg_to_join"         # Predicted segments
noisy_dir = "data/seg_to_join_noisy"  # Noisy segments

# Output directories
os.makedirs("data/full_tracks", exist_ok=True)
os.makedirs("data/full_tracks_noisy", exist_ok=True)
os.makedirs("data/full_tracks_pred", exist_ok=True)

# Function to group segments by track ID (e.g., 'audio00010')
def group_segments_by_track(files):
    track_dict = defaultdict(list)
    for f in files:
        basename = os.path.basename(f)
        try:
            base = basename.split("_segment_")
            track_id = base[0]  # e.g., 'audio00010'
            # Extract segment number (digits only)
            seg_num_str = ''.join(c for c in base[1] if c.isdigit())
            segment_num = int(seg_num_str)
            track_dict[track_id].append((segment_num, f))
        except Exception as e:
            print(f"Unexpected filename: {basename} -> Error: {e}")
    return track_dict

# Function to process a folder of segments and save concatenated tracks
def process_folder(folder_path, output_path):
    all_files = glob(os.path.join(folder_path, "*.wav"))
    grouped = group_segments_by_track(all_files)

    for track_id, segments in grouped.items():
        # Sort segments by segment number
        segments.sort()
        waveforms = []
        sample_rate = None

        for seg_num, path in segments:
            waveform, sr = torchaudio.load(path)
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                raise ValueError(f"Incompatible sample rate in {path}")
            waveforms.append(waveform)

        # Concatenate all segments along time dimension (dim=1)
        full_track = torch.cat(waveforms, dim=1)

        # Save full track
        out_path = os.path.join(output_path, f"{track_id}.wav")
        torchaudio.save(out_path, full_track, sample_rate)
        print(f"Saved: {out_path}")

# Execute for predicted, noisy, and original segments
process_folder(pred_dir, "data/full_tracks_pred")
process_folder(noisy_dir, "data/full_tracks_noisy")
process_folder(og_dir, "data/full_tracks")
