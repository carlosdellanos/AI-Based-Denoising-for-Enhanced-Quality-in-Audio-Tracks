import os
import librosa
import soundfile as sf
import shutil
from pydub import AudioSegment

# This script preprocesses audio data for training, validation, and testing.
# It performs the following operations:
#   1. Converts all audio files into .wav format (if not already).
#   2. Splits long audio tracks into smaller segments of fixed duration.
#   3. Extends short audio tracks by repeating them until they reach the target duration.
#
# Parameters:
#   TARGET_SR (int): Desired sampling rate in Hz.
#   TARGET_DURATION (int): Desired length of each processed track in seconds.
#
# The final dataset is stored in the "data/split_data" folder with three subfolders:
# "train", "val", and "test", where each audio file is converted, segmented, and extended.

TARGET_SR = 22050     # Desired sampling frequency in Hz
TARGET_DURATION = 2   # Desired track length in seconds


# Converting the track files into .wav
def convert_to_wav(input_path, output_path):
    # Load the file
    audio = AudioSegment.from_file(input_path)
    
    # Export the file to .wav format
    audio.export(output_path, format="wav")
    print(f"File converted and saved as: {output_path}")


def split_audio(input_path, output_folder, segment_duration_ms):
    # Load the audio track using AudioSegment
    audio = AudioSegment.from_file(input_path)
    
    # Get track duration in ms
    track_duration_ms = len(audio)

    # Tracks shorter than the segment length are only renamed
    if track_duration_ms <= segment_duration_ms + 1000:
        base_name = os.path.splitext(os.path.basename(input_path))[0]   # File name without extension
        new_filename = f"{base_name}_whole_audio.wav"
        new_path = os.path.join(output_folder, new_filename)

        os.rename(input_path, new_path)  # Rename the file

        print(f"Renamed (not divided): {new_path}")
        return

    # Split the track into segments of specified length (segment_duration_ms)
    for start_ms in range(0, track_duration_ms, segment_duration_ms):
        end_ms = min(start_ms + segment_duration_ms, track_duration_ms)  # Determine exact end point of segment
        segment = audio[start_ms:end_ms]
        
        # Save each segment of the track with its own name
        base_name = os.path.splitext(os.path.basename(input_path))[0]  
        segment_filename = os.path.join(output_folder, f"{base_name}_segment_{start_ms // segment_duration_ms + 1}.wav")
        segment.export(segment_filename, format="wav")
        print(f"Saved as: {segment_filename}")
    
    # Remove the original unsplit file 
    os.remove(input_path)
    print(f"Original file removed: {input_path}")


def extend_audio_to_length(input_path, min_length_ms):
    # Load the audio track
    audio = AudioSegment.from_file(input_path)

    # If the audio track is shorter than the desired length, repeat it until reaching the target length
    if len(audio) < min_length_ms:
        repeated_segments = []
        while len(audio) < min_length_ms:
            repeated_segments.append(audio) 
            audio = sum(repeated_segments)    

        adjusted_audio = audio[:min_length_ms]
    
    # If the audio track is longer than the desired length, trim it
    elif len(audio) > min_length_ms:
        adjusted_audio = audio[:min_length_ms]
        
    else:
        # No extension needed
        return
        

    # Set the new output path
    output_path_extended = input_path.replace(".wav", "_adjusted.wav")

    # Save the extended file
    adjusted_audio.export(output_path_extended, format="wav")
    print(f"Extended file saved: {output_path_extended}")

    # Remove the original file
    os.remove(input_path)


# Process each split (train, val, test)
base_folder = "data/split_data"
splits = ["train", "val", "test"]

for split in splits:
    split_folder = os.path.join(base_folder, split)
    
    # Iterate over the folder with raw files and apply preprocessing
    for filename in os.listdir(split_folder):
        input_path = os.path.join(split_folder, filename)
    
        # Generate output path for the .wav file 
        wav_filename = f"{os.path.splitext(filename)[0]}.wav"
        output_wav_path = os.path.join(split_folder, wav_filename)

        # Skip already processed files
        if os.path.exists(output_wav_path):
            continue

        # Convert the file into .wav if needed
        if not filename.endswith(".wav"):
            convert_to_wav(input_path, output_wav_path)
    
        # If file is already a .wav, copy it to the output path
        else:
            shutil.copy2(input_path, output_wav_path)
            print(f".wav file copied to split_folder: {output_wav_path}")

        # Split long files into segments of TARGET_DURATION length
        split_audio(output_wav_path, split_folder, segment_duration_ms=TARGET_DURATION * 1000)

    # Extend files that are too short (shorter than TARGET_DURATION)     
    for filename in os.listdir(split_folder):
        input_path = os.path.join(split_folder, filename)
        extend_audio_to_length(input_path, min_length_ms=TARGET_DURATION * 1000)
        print("File processed successfully")
