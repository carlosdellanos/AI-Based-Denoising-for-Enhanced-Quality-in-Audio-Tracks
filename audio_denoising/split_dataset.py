# VERSION FOR RENAMING AND SPLITTING DATASET
# IMPORTANT: This script is intended only for the following noise classes:
# AddWhiteNoise, AddQuantizationNoise, AddQuantizationNoiseWithDither
# It is NOT intended for AddRecordedNoise

import os
import shutil
from sklearn.model_selection import train_test_split

def split_and_rename_dataset(raw_folder, output_base_folder, seed=42):
    # Create output folders
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_base_folder, split), exist_ok=True)

    # Get and sort files
    raw_files = sorted([
        f for f in os.listdir(raw_folder)
        if os.path.isfile(os.path.join(raw_folder, f))
    ])

    # Rename files: keep a list of (old_name, new_name)
    renamed_files = []
    for idx, old_name in enumerate(raw_files):
        ext = os.path.splitext(old_name)[1]
        new_name = f"audio{idx+1:05d}{ext}"
        renamed_files.append((old_name, new_name))

    new_names = [new_name for _, new_name in renamed_files]

    # Split dataset
    train_files, temp_files = train_test_split(new_names, test_size=0.3333, random_state=seed)
    val_files, test_files = train_test_split(temp_files, test_size=0.6, random_state=seed)

    # Map new_name to original name for copying
    name_map = {new_name: old_name for old_name, new_name in renamed_files}

    # Copy and rename files into splits
    for split, file_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for new_name in file_list:
            old_name = name_map[new_name]
            src = os.path.join(raw_folder, old_name)
            dst = os.path.join(output_base_folder, split, new_name)
            shutil.copy2(src, dst)

    print("Dataset renamed and split into train/val/test successfully.")

# Example usage
split_and_rename_dataset("data/raw_data", "data/split_data")
