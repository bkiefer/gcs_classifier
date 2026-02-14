import soundfile as sf
import numpy as np 
import torch
import librosa

def load_audio(file_path):
    """
    Loads an audio file from the given file path.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        tuple: A tuple containing:
            - waveform (numpy.ndarray): The audio data as a NumPy array.
            - sampling_rate (int): The sampling rate of the audio.
        Returns (None, None) if an error occurs (e.g., file not found).
    """
    try:
        waveform, sampling_rate = sf.read(file_path)
        print(f"Successfully loaded {file_path}")
        print(f"Shape of waveform: {waveform.shape}, Sampling rate: {sampling_rate} Hz")
        return waveform, sampling_rate
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None


def initialize_vad_model():
    """
    Initializes and returns the Silero VAD model and its utility functions.

    Returns:
        tuple: A tuple containing:
            - model: The loaded VAD model.
            - utils: A dictionary of utility functions from the Silero repo.
    """
    # This downloads the Silero VAD model from PyTorch Hub
    # The model is cached after the first download.
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False) # Set to True to re-download
        print("Silero VAD model initialized successfully.")
        return model, utils
    except Exception as e:
        print(f"Error initializing VAD model: {e}")
        return None, None

def get_all_sound_events(audio_waveform, sampling_rate, top_db=25, min_duration_s=0.2):
    """
    Finds all non-silent audio events in a file using a simple energy-based VAD.
    This is our primary, sensitive segmenter.
    """
    # Resample to 16k if needed, as most subsequent steps expect it
    if sampling_rate != 16000:
        audio_waveform = librosa.resample(y=audio_waveform, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000

    clips = librosa.effects.split(y=audio_waveform, top_db=top_db)
    
    timestamps = []
    for start_sample, end_sample in clips:
        start_sec = start_sample / sampling_rate
        end_sec = end_sample / sampling_rate
        if (end_sec - start_sec) >= min_duration_s:
            timestamps.append({'start': start_sec, 'end': end_sec})
            
    return timestamps, audio_waveform, sampling_rate

def get_speech_prob(audio_chunk_16k, vad_model):
    """
    Calculates the average speech probability for a given audio chunk.
    It breaks the chunk into smaller, model-compatible pieces and averages the results.
    """
    if vad_model is None:
        return 0.0
    
    # The model strictly requires 512 sample chunks for 16kHz audio.
    chunk_size = 512
    
    # If the chunk is smaller than the required size, pad it with silence
    if len(audio_chunk_16k) < chunk_size:
        padding = torch.zeros(chunk_size - len(audio_chunk_16k), dtype=torch.float32)
        audio_chunk_16k = torch.cat([torch.from_numpy(audio_chunk_16k).float(), padding])
    else:
        audio_chunk_16k = torch.from_numpy(audio_chunk_16k).float()

    # List to store probabilities of each small chunk
    speech_probs = []
    
    # Iterate through the audio chunk in windows of the required size
    for i in range(0, len(audio_chunk_16k), chunk_size):
        chunk = audio_chunk_16k[i : i + chunk_size]
        
        # If the last chunk is smaller, pad it
        if len(chunk) < chunk_size:
            padding = torch.zeros(chunk_size - len(chunk))
            chunk = torch.cat([chunk, padding])

        # Get the speech probability for the small chunk
        speech_prob = vad_model(chunk.unsqueeze(0), 16000).item()
        speech_probs.append(speech_prob)
    
    # Return the average probability over all the small chunks
    if not speech_probs:
        return 0.0
    
    return sum(speech_probs) / len(speech_probs)

if __name__ == "__main__":
    
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    sample_audio_path = os.path.join(project_root, "data/validation_set/gcs_2", "1-36400-A-23.wav") 
    
    print(f"--- Testing audio_utils.py ---")
    print(f"Attempting to load: {sample_audio_path}")

    data_folder = os.path.join(project_root, "data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(sample_audio_path):
        print(f"Please place a WAV file at {sample_audio_path} for testing.")
    else:
        # 1. Load the audio
        waveform_data, sr = load_audio(sample_audio_path)

        if waveform_data is not None:
            print("\n--- Testing Energy-Based Event Detection ---")
            # 2. Get all sound events using the new energy-based function
            sound_events, waveform_16k, sr_16k = get_all_sound_events(waveform_data, sr)

            if sound_events:
                print(f"Found {len(sound_events)} potential sound events:")
                for i, ts in enumerate(sound_events):
                    print(f"  Event {i+1}: Start={ts['start']:.2f}s, End={ts['end']:.2f}s")
            else:
                print("No sound events detected.")

            print("\n--- Testing Silero VAD Speech Probability ---")
            # 3. Initialize the VAD model to test the probability function
            vad_model_instance, _ = initialize_vad_model() # We don't need utils here

            if vad_model_instance and sound_events:
                # Let's test the speech probability of the first detected event
                first_event = sound_events[0]
                start_sample = int(first_event['start'] * sr_16k)
                end_sample = int(first_event['end'] * sr_16k)
                audio_chunk = waveform_16k[start_sample:end_sample]

                if len(audio_chunk) > 0:
                    speech_prob = get_speech_prob(audio_chunk, vad_model_instance)
                    print(f"Speech probability for the first event: {speech_prob:.4f}")
                    if speech_prob > 0.5:
                        print("  -> This event is likely speech.")
                    else:
                        print("  -> This event is likely not speech.")
                else:
                    print("First event audio chunk is empty, cannot test speech probability.")
            elif not sound_events:
                 print("Cannot test speech probability as no events were detected.")
            else:
                print("Failed to initialize VAD model.")

        else:
            print("Failed to load audio, cannot perform tests.")