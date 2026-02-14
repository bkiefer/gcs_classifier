import os
import sys
import pandas as pd
import time
from glob import glob

# Setup Project Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import Pipeline Components
from src.audio_utils import load_audio, initialize_vad_model,  get_all_sound_events, get_speech_prob
from src.transcription import initialize_whisper_model, transcribe_chunk
from src.feature_extractor import (initialize_dictionary, calculate_lexical_validity,
                                   initialize_language_model, calculate_perplexity)

def find_audio_files(directory):
    """Finds all WAV audio files in a given directory and its subdirectories."""
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files

def process_single_file_for_features(file_path, models):
    """
    Processes a single audio file with the robust two-stage filtering logic.
    """
    # Unpack the models dictionary for easier access
    vad_model, _ = models['vad'] # We only need the model for get_speech_prob
    whisper_model = models['whisper']
    german_dict = models['dictionary']
    lm_tokenizer, lm_model = models['lm']

    all_results = []
    
    # 1. Load audio with its original sampling rate
    waveform_orig, sampling_rate_orig = load_audio(file_path)
    if waveform_orig is None:
        return []

    # 2. Sensitive First Pass: Get ALL potential sound events using energy-based VAD.
    #    This also handles resampling the waveform to 16kHz for all subsequent steps.
    sound_events, waveform_16k, sr_16k = get_all_sound_events(waveform_orig, sampling_rate_orig)
    
    if not sound_events:
        return [] # File is likely pure silence, skip it.

    # 3. Process and FILTER each potential sound event
    for i, segment in enumerate(sound_events):
        start_sec = segment['start']
        end_sec = segment['end']
        
        start_sample = int(start_sec * sr_16k)
        end_sample = int(end_sec * sr_16k)
        audio_chunk = waveform_16k[start_sample:end_sample]

        if len(audio_chunk) < 100: # Skip extremely short, probably junk segments
            continue

        # --- SMART FILTERING STAGE ---
        # A. Get Silero VAD's opinion on the chunk
        silero_speech_prob = get_speech_prob(audio_chunk, vad_model)
        
        # B. Get Whisper's opinion by transcribing
        transcription_result = transcribe_chunk(audio_chunk, whisper_model, language="de")
        
        # C. Extract Whisper's no-speech probability
        whisper_no_speech_prob = transcription_result.get('no_speech_prob', 1.0) if transcription_result else 1.0

        # --- DECISION LOGIC ---
        # Accept the segment if Silero has some confidence OR Whisper is confident it IS speech.
        # These thresholds (0.25 and 0.60) are tunable parameters!
        is_human_vocalization = (silero_speech_prob > 0.25) or (whisper_no_speech_prob < 0.60)

        if not is_human_vocalization:
            # This segment is likely background noise (GCS 1 with noise)
            print(f"      -> Skipping segment ({start_sec:.2f}s-{end_sec:.2f}s). "
                  f"Filtered as noise (Silero P={silero_speech_prob:.2f}, Whisper NoSpeechP={whisper_no_speech_prob:.2f}).")
            continue # Skip to the next sound event

        # --- FEATURE EXTRACTION (only runs if the segment passed the filter) ---
        if transcription_result and transcription_result['text']:
            transcript_text = transcription_result['text']
            
            features = {
                'avg_logprob': transcription_result['avg_logprob'],
                'lexical_validity': calculate_lexical_validity(transcript_text, german_dict),
                'perplexity': calculate_perplexity(transcript_text, lm_tokenizer, lm_model)
            }
            
            # This is the final dictionary for our CSV row
            final_result = {
                'filepath': file_path,
                'start_time': start_sec,
                'end_time': end_sec,
                'transcription': transcript_text,
                'silero_speech_prob': silero_speech_prob, # Saving this for analysis is very useful!
                'whisper_no_speech_prob': whisper_no_speech_prob, # Also very useful
                **features
            }
            all_results.append(final_result)
            
    return all_results

def run_batch_for_set(set_name, data_root, models):
    """
    Runs the full batch processing for a given dataset split (e.g., 'validation_set').
    """
    print(f"\n{'='*20} STARTING BATCH PROCESSING FOR: {set_name.upper()} {'='*20}")

    set_path = os.path.join(data_root, set_name)
    data_categories = {
        "gcs_1": os.path.join(set_path, "gcs_1"),
        "gcs_2": os.path.join(set_path, "gcs_2"),
        "gcs_3": os.path.join(set_path, "gcs_3"),
        "gcs_45_clean": os.path.join(set_path, "gcs_45"),
        "gcs_45_noisy": os.path.join(set_path, "gcs_45_noisy"),
    }
        
    all_feature_results = []
    
    for category_name, category_path in data_categories.items():
        print(f"\n--- Processing category: {category_name} ---")
        print(f"Searching for audio files in: {category_path}")
        
        audio_files_to_process = []
        supported_extensions = ['*.wav', '*.mp3']
        for ext in supported_extensions:
            audio_files_to_process.extend(glob(os.path.join(category_path, ext)))

        if not audio_files_to_process:
            print(f"Warning: No audio files found in '{category_path}'. Skipping.")
            continue
            
        print(f"Found {len(audio_files_to_process)} files to process.")
        
        for i, file_path in enumerate(audio_files_to_process):
            print(f"  -> Processing file {i+1}/{len(audio_files_to_process)}: {os.path.basename(file_path)}")

            file_results = process_single_file_for_features(file_path, models)

            for result in file_results:
                result['category'] = category_name
                if category_name == "gcs_45_noisy" and "snr" in os.path.basename(file_path):
                    try:
                        snr_str = os.path.basename(file_path).split('snr')[1].split('.')[0]
                        result['snr'] = int(snr_str)
                    except (IndexError, ValueError):
                        result['snr'] = None
                else:
                    result['snr'] = None
            
            all_feature_results.extend(file_results)

    return all_feature_results


if __name__ == "__main__":
    # --- Configuration ---
    WHISPER_MODEL_SIZE = "base"
    LM_MODEL_ID = "dbmdz/german-gpt2"
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # --- Initialization (Done once for the whole script) ---
    print("--- Initializing all models for batch processing ---")
    start_init_time = time.time()
    models_dict = {
        'vad': initialize_vad_model(),
        'whisper': initialize_whisper_model(model_size=WHISPER_MODEL_SIZE),
        'dictionary': initialize_dictionary(),
        'lm': initialize_language_model(model_id=LM_MODEL_ID)
    }
    end_init_time = time.time()
    print(f"--- Models initialized in {end_init_time - start_init_time:.2f} seconds ---\n")

    if not all(m is not None for v in models_dict.values() for m in (v if isinstance(v, tuple) else [v])):
        print("A model failed to initialize. Aborting batch processing.")
    else:
        # --- Main Batch Processing Logic ---
        total_start_time = time.time()
        
        # Process both validation and test sets
        for data_set_name in ["validation_set", "test_set"]:
            results_for_set = run_batch_for_set(data_set_name, data_root, models_dict)
            
            if results_for_set:
                output_csv_path = os.path.join(results_dir, f"features_{data_set_name}.csv")
                print(f"\nSaving {len(results_for_set)} total segments from {data_set_name} to CSV: {output_csv_path}")
                
                df = pd.DataFrame(results_for_set)
                column_order = ['filepath', 'category', 'snr', 'start_time', 'end_time', 
                                'transcription', 'avg_logprob', 'lexical_validity', 'perplexity', 'silero_speech_prob', 'whisper_no_speech_prob']

                df = df.reindex(columns=column_order)
                df.to_csv(output_csv_path, index=False, encoding='utf-8')
                print(f"Successfully saved {data_set_name} feature dataset.")
            else:
                print(f"No results were generated for {data_set_name}. CSV file not saved.")

        total_end_time = time.time()
        print(f"\n--- Total batch processing complete in {(total_end_time - total_start_time)/60:.2f} minutes ---")