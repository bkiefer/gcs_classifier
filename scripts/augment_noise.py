import os
import soundfile as sf
import numpy as np
import random
import librosa 
from glob import glob

def calculate_rms(audio):
    """Calculates the Root Mean Square of an audio signal."""
    return np.sqrt(np.mean(audio**2))

def add_noise(clean_audio, noise_audio, snr_db):
    """
    Adds noise to a clean audio signal at a specific SNR level.

    Args:
        clean_audio (numpy.ndarray): The clean audio waveform.
        noise_audio (numpy.ndarray): The noise audio waveform.
        snr_db (int): The desired Signal-to-Noise Ratio in dB.

    Returns:
        numpy.ndarray: The noisy audio waveform.
    """
    rms_clean = calculate_rms(clean_audio)
    
    if rms_clean == 0:
        return clean_audio

    # Make noise segment the same length as the clean audio
    if len(clean_audio) >= len(noise_audio):
        repeat_factor = int(np.ceil(len(clean_audio) / len(noise_audio)))
        noise_segment = np.tile(noise_audio, repeat_factor)[:len(clean_audio)]
    else:
        # Take a random snippet of noise if it's longer
        start_index = random.randint(0, len(noise_audio) - len(clean_audio))
        noise_segment = noise_audio[start_index : start_index + len(clean_audio)]

    rms_noise = calculate_rms(noise_segment)
    
    if rms_noise == 0:
        return clean_audio

    # Calculate the required scaling factor for the noise
    snr_linear = 10**(snr_db / 20.0)
    required_rms_noise = rms_clean / snr_linear
    scaling_factor = required_rms_noise / rms_noise

    # Scale the noise and add it to the clean audio
    noisy_audio = clean_audio + (noise_segment * scaling_factor)
    return noisy_audio


def process_set(set_name, base_dir, noise_files, target_snrs_db):
    """
    Processes a whole set (e.g., 'validation_set' or 'test_set').
    """
    print(f"\n--- Processing {set_name} ---")
    
    clean_speech_dir = os.path.join(base_dir, set_name, "gcs_45")
    output_dir = os.path.join(base_dir, set_name, "gcs_45_noisy")
    os.makedirs(output_dir, exist_ok=True)

    try:
        clean_files = glob(os.path.join(clean_speech_dir, '*.*')) # Using glob to find any audio file (maybe find a more elegant solution)
    except FileNotFoundError:
        print(f"Error: Directory not found: {clean_speech_dir}. Skipping this set.")
        return

    if not clean_files:
        print(f"No clean speech files found in {clean_speech_dir}. Skipping.")
        return
        
    print(f"Found {len(clean_files)} clean files in {set_name}. Starting augmentation...")
    
    for i, clean_filepath in enumerate(clean_files):
        clean_filename = os.path.basename(clean_filepath)
        print(f"  -> Augmenting file {i+1}/{len(clean_files)}: '{clean_filename}'")
        
        try:
            clean_audio, sr = librosa.load(clean_filepath, sr=16000, mono=True)
        except Exception as e:
            print(f"    Could not load {clean_filename}, skipping. Error: {e}")
            continue

        noise_filepath = random.choice(noise_files)
        noise_filename = os.path.basename(noise_filepath)
        
        try:
            noise_audio, _ = librosa.load(noise_filepath, sr=16000, mono=True)
        except Exception as e:
            print(f"    Could not load {noise_filename}, skipping. Error: {e}")
            continue

        for snr in target_snrs_db:
            noisy_audio = add_noise(clean_audio, noise_audio, snr)
            
            base_name = os.path.splitext(clean_filename)[0]
            noise_base_name = os.path.splitext(noise_filename)[0]
            output_filename = f"{base_name}_noise_{noise_base_name}_snr{snr}.wav"
            output_filepath = os.path.join(output_dir, output_filename)
            
            sf.write(output_filepath, noisy_audio, sr)

    print(f"--- Finished processing {set_name} ---")


if __name__ == "__main__":
    print("--- Starting Noise Augmentation Script ---")

    # --- Configuration ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    data_root = os.path.join(project_root, "data")
    noise_dir = os.path.join(data_root, "noise_samples")
    
    # Desired SNR levels to generate
    target_snrs_db = [15, 10, 5, 0, -5]

    # --- Main Logic ---
    try:
            all_noise_files = glob(os.path.join(noise_dir, '*.wav'))
            if not all_noise_files:
                print(f"Error: No noise sample WAV files found in: {noise_dir}")
                print("Please add noise files to continue.")
            else:
                print(f"Found {len(all_noise_files)} noise files.")
                # Process both the validation set and the test set
                process_set("validation_set", data_root, all_noise_files, target_snrs_db)
                process_set("test_set", data_root, all_noise_files, target_snrs_db)
                print("\n--- All noise augmentation complete! ---")
    except FileNotFoundError:
            print(f"Error: Noise directory not found at {noise_dir}")