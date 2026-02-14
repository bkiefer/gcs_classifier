# A single, self-contained script to run the real-time performance simulation for multiple Whisper model sizes sequentially.

import time
import os
import torch
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import string
import enchant
from transformers import AutoTokenizer, AutoModelForCausalLM
from faster_whisper import WhisperModel

# --- All Helper Functions are now in this one file ---

# === From audio_utils.py ===
def load_audio(file_path):
    try:
        waveform, sampling_rate = sf.read(file_path, dtype='float32')
        # Ensure mono by taking the mean if stereo
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        return waveform, sampling_rate
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None

def initialize_vad_model():
    try:
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        return model
    except Exception as e:
        print(f"Error initializing VAD model: {e}")
        return None

# === From transcription.py ===
def initialize_whisper_model(model_size, device="cpu", compute_type="int8"):
    print(f"Initializing FasterWhisper model '{model_size}'...")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"Model '{model_size}' initialized successfully.")
        return model
    except Exception as e:
        print(f"Error initializing Whisper model '{model_size}': {e}")
        return None

def transcribe_chunk(audio_chunk_waveform, whisper_model, language="de"):
    try:
        processing_start_time = time.perf_counter()
        segments, _ = whisper_model.transcribe(audio_chunk_waveform, language=language)
        segment_results = list(segments)
        processing_end_time = time.perf_counter()
        
        result = {
            'text': '', 
            'avg_logprob': -3.0, 
            'no_speech_prob': 1.0,
            'transcription_time_s': processing_end_time - processing_start_time
        }

        if segment_results:
            first_segment = segment_results[0]
            result['text'] = first_segment.text.strip()
            result['avg_logprob'] = first_segment.avg_logprob
            result['no_speech_prob'] = first_segment.no_speech_prob
        
        return result
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

# === From feature_extractor.py ===
def initialize_dictionary(lang_code="de_DE"):
    try:
        return enchant.Dict(lang_code)
    except enchant.errors.DictNotFoundError:
        print(f"Error: Dictionary for '{lang_code}' not found. Please install hunspell-de-de or similar.")
        return None

def initialize_language_model(model_id="dbmdz/german-gpt2"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        return tokenizer, model
    except Exception as e:
        print(f"Error initializing Hugging Face model: {e}")
        return None, None

def calculate_lexical_validity(text, german_dictionary):
    if not text or not german_dictionary: return 0.0
    translator = str.maketrans('', '', string.punctuation)
    words = text.lower().translate(translator).split()
    if not words: return 0.0
    valid_word_count = sum(1 for word in words if german_dictionary.check(word))
    return valid_word_count / len(words)

def calculate_perplexity(text, tokenizer, model):
    if not text: return float('inf')
    try:
        inputs = tokenizer(text, return_tensors="pt")
        if inputs.input_ids.size(1) == 0: return float('inf')
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()
    except Exception:
        return float('inf')

# === From classifier.py ===
def normalize_feature(value, min_val, max_val, invert=False):
    clamped_value = np.clip(value, min_val, max_val)
    normalized = (clamped_value - min_val) / (max_val - min_val)
    return 1.0 - normalized if invert else normalized

def classify_gcs_level_rule_based(normalized_features, no_speech_prob, raw_features, thresholds):
    if no_speech_prob > 0.8: return 1
    num_words = len(str(raw_features.get('transcription', '')).split())

    if num_words <= 2:
        return 5 if normalized_features['norm_confidence'] > thresholds['conf_short'] else 2
    
    if normalized_features['norm_lexical'] < thresholds['lex_gcs2'] or \
       normalized_features['norm_confidence'] < thresholds['conf_gcs2']:
        return 2
        
    if normalized_features['norm_inv_perplexity'] > thresholds['ppl_gcs45'] and \
       normalized_features['norm_confidence'] > thresholds['conf_gcs45']:
        return 5
        
    return 3

# === Main Simulation Logic ===
def simulate_realtime_processing(file_path, models):
    try:
        from silero_vad.utils_vad import VADIterator
    except ImportError:
        print("Could not import VADIterator. Ensure Silero VAD is installed.")
        return None

    print(f"\n--- Running REAL-TIME SIMULATION for: {os.path.basename(file_path)} ---")
    
    waveform, sampling_rate = load_audio(file_path)
    if waveform is None: return None
    
    if sampling_rate != 16000:
        waveform = librosa.resample(y=waveform, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000
    
    threshold_params = {
        'conf_short': 0.6, 'lex_gcs2': 0.5, 'conf_gcs2': 0.45,
        'ppl_gcs45': 0.75, 'conf_gcs45': 0.55
    }
    
    window_size_samples = 512
    vad_iterator = VADIterator(models['vad'], threshold=0.5, sampling_rate=sampling_rate)
    
    all_metrics = []
    speech_start_time = None
    
    for i in range(0, len(waveform), window_size_samples):
        chunk = waveform[i : i + window_size_samples]
        if len(chunk) < window_size_samples: break
            
        speech_dict = vad_iterator(chunk, return_seconds=True)
        if speech_dict:
            if 'start' in speech_dict:
                speech_start_time = speech_dict['start']
            
            if 'end' in speech_dict and speech_start_time is not None:
                speech_end_time = speech_dict['end']
                segment_end_time_in_stream = (i + window_size_samples) / sampling_rate
                
                pipeline_start_time = time.perf_counter()
                
                start_sample, end_sample = int(speech_start_time * sampling_rate), int(speech_end_time * sampling_rate)
                speech_chunk_waveform = waveform[start_sample:end_sample]
                speech_duration_s = len(speech_chunk_waveform) / sampling_rate

                if speech_duration_s < 0.2:
                    speech_start_time = None
                    continue

                transcription_result = transcribe_chunk(speech_chunk_waveform, models['whisper'])
                
                if transcription_result and transcription_result.get('text'):
                    pass
                
                pipeline_end_time = time.perf_counter()
                
                processing_time = pipeline_end_time - pipeline_start_time
                latency = (processing_time) + (segment_end_time_in_stream - speech_end_time)
                rtf = processing_time / speech_duration_s if speech_duration_s > 0 else float('inf')
                      
                all_metrics.append({'latency_s': latency, 'rtf': rtf, 'duration_s': speech_duration_s})
                speech_start_time = None
    
    vad_iterator.reset_states()
    
    if all_metrics:
        avg_latency = np.mean([m['latency_s'] for m in all_metrics])
        avg_rtf = np.mean([m['rtf'] for m in all_metrics])
        return {'avg_latency': avg_latency, 'avg_rtf': avg_rtf}
    else:
        return None

if __name__ == "__main__":
    # --- Configuration ---
    MODELS_TO_TEST = ["small", "base", "medium"]
    LM_MODEL_ID = "dbmdz/german-gpt2"
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Path to a sample audio file for testing
    input_audio_path = os.path.join(project_root, "data", "sample4.wav") 
    
    print("--- Initializing shared models (VAD, Dictionary, LM) ---")
    shared_models = {
        'vad': initialize_vad_model(),
        'dictionary': initialize_dictionary(),
        'lm_tokenizer': None,
        'lm_model': None
    }
    shared_models['lm_tokenizer'], shared_models['lm_model'] = initialize_language_model(model_id=LM_MODEL_ID)
    
    if not all(shared_models.values()):
        print("\nOne or more shared models failed to initialize. Aborting.")
    else:
        # --- Main Loop to Test Each Whisper Model ---
        all_run_results = []
        
        for model_size in MODELS_TO_TEST:
            print(f"\n{'='*20} TESTING MODEL: {model_size.upper()} {'='*20}")
            
            # Initialize the specific Whisper model for this run
            whisper_model = initialize_whisper_model(model_size=model_size)
            
            if whisper_model:
                current_models = shared_models.copy()
                current_models['whisper'] = whisper_model
                
                # Run the simulation
                metrics = simulate_realtime_processing(input_audio_path, current_models)
                
                if metrics:
                    metrics['model_size'] = model_size
                    all_run_results.append(metrics)
                
                # Clean up memory before loading the next model apparently important for large models
                del whisper_model
                del current_models
                if torch.cuda.is_available(): 
                    torch.cuda.empty_cache()

        # --- Final Report ---
        if all_run_results:
            print("\n\n" + "="*20 + " FINAL PERFORMANCE REPORT " + "="*20)
            results_df = pd.DataFrame(all_run_results)
            results_df = results_df[['model_size', 'avg_latency', 'avg_rtf']]
            results_df.rename(columns={
                'model_size': 'Model Size',
                'avg_latency': 'Avg. Latency (s)',
                'avg_rtf': 'Avg. Real-Time Factor (RTF)'
            }, inplace=True)
            
            print(results_df.to_string(index=False))
            print("\n(RTF < 1.0 means faster than real-time)")
        else:
            print("\nNo simulations were successfully completed.")