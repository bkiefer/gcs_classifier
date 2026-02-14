import os
import random
import shutil
from glob import glob

def split_files(file_list, validation_ratio=0.7):
    """
    Shuffles a list of files and splits it into two lists based on the ratio.
    """
    random.shuffle(file_list)
    split_index = int(len(file_list) * validation_ratio)
    validation_files = file_list[:split_index]
    test_files = file_list[split_index:]
    return validation_files, test_files

def copy_files(file_list, destination_dir):
    """
    Copies a list of files to a destination directory.
    """
    os.makedirs(destination_dir, exist_ok=True)
    for file_path in file_list:
        try:
            shutil.copy(file_path, destination_dir)
        except Exception as e:
            print(f"Error copying {file_path} to {destination_dir}: {e}")

if __name__ == "__main__":
    print("--- Starting Dataset Preparation and Splitting Script ---")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, "data")
    
    # SOURCE directories
    source_gcs2 = os.path.join(data_root, "gcs_level_2_incomprehensible")
    source_gcs3 = os.path.join(data_root, "gcs_level_3_word_salad")
    source_gcs45_full = os.path.join(data_root, "CommonVoice21.0/cv-corpus-21.0-2025-03-14/de/clips")
    
    # DESTINATION directories for the split data
    validation_dir = os.path.join(data_root, "validation_set")
    test_dir = os.path.join(data_root, "test_set")

    # GCS 4/5 (Clean Speech) is to big, therefore, only a random selection
    # 200 files (maybe needs adjustment)
    NUM_CLEAN_FILES_TO_USE = 200
    VALIDATION_SPLIT_RATIO = 0.7

    # --- Main Logic ---

    # 1. Handle GCS 4/5 (Clean Speech)
    print(f"\n1. Processing GCS 4/5 (Clean Speech) from: {source_gcs45_full}")
    try:
        all_clean_files = glob(os.path.join(source_gcs45_full, '*.mp3')) 
        if len(all_clean_files) < NUM_CLEAN_FILES_TO_USE:
            print(f"Warning: Only found {len(all_clean_files)} clean files, using all of them.")
            selected_clean_files = all_clean_files
        else:
            selected_clean_files = random.sample(all_clean_files, NUM_CLEAN_FILES_TO_USE)
            print(f"Randomly selected {NUM_CLEAN_FILES_TO_USE} clean files to use.")

        gcs45_val, gcs45_test = split_files(selected_clean_files, VALIDATION_SPLIT_RATIO)
        
        # Copy files to their new homes
        print(f"Copying {len(gcs45_val)} files to validation_set/gcs_45...")
        copy_files(gcs45_val, os.path.join(validation_dir, "gcs_45"))
        print(f"Copying {len(gcs45_test)} files to test_set/gcs_45...")
        copy_files(gcs45_test, os.path.join(test_dir, "gcs_45"))
    except FileNotFoundError:
        print(f"Error: Directory not found: {source_gcs45_full}. Please check the path.")

    # 2. Handle GCS 3 (Word Salad)
    print(f"\n2. Processing GCS 3 (Word Salad) from: {source_gcs3}")
    try:
        all_gcs3_files = glob(os.path.join(source_gcs3, '*.wav'))
        gcs3_val, gcs3_test = split_files(all_gcs3_files, VALIDATION_SPLIT_RATIO)

        print(f"Copying {len(gcs3_val)} files to validation_set/gcs_3...")
        copy_files(gcs3_val, os.path.join(validation_dir, "gcs_3"))
        print(f"Copying {len(gcs3_test)} files to test_set/gcs_3...")
        copy_files(gcs3_test, os.path.join(test_dir, "gcs_3"))
    except FileNotFoundError:
         print(f"Error: Directory not found: {source_gcs3}. Please check the path.")

    # 3. Handle GCS 2 (Incomprehensible)
    print(f"\n3. Processing GCS 2 (Incomprehensible) from: {source_gcs2}")
    try:
        all_gcs2_files = glob(os.path.join(source_gcs2, '*.wav'))
        gcs2_val, gcs2_test = split_files(all_gcs2_files, VALIDATION_SPLIT_RATIO)
        
        print(f"Copying {len(gcs2_val)} files to validation_set/gcs_2...")
        copy_files(gcs2_val, os.path.join(validation_dir, "gcs_2"))
        print(f"Copying {len(gcs2_test)} files to test_set/gcs_2...")
        copy_files(gcs2_test, os.path.join(test_dir, "gcs_2"))
    except FileNotFoundError:
         print(f"Error: Directory not found: {source_gcs2}. Please check the path.")

    print("\n--- Dataset preparation and splitting complete! ---")
    print(f"Your validation and test sets are ready in the '{validation_dir}' and '{test_dir}' folders.")