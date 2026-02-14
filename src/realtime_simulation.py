import time
import os
import torch
import numpy as np

from audio_utils import load_audio
from transcription import initialize_whisper_model, transcribe_chunk
from feature_extractor import (initialize_dictionary, calculate_lexical_validity,
                               initialize_language_model, calculate_perplexity)
from classifier import normalize_feature, classify_gcs_level_rule_based

# Import written by Copilot
# Import in a way that doesn't require a separate initialization function
try:
    torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    from silero_vad.utils_vad import VADIterator
except Exception as e:
    print(f"Could not load VADIterator. Ensure Silero VAD is installed. Error: {e}")
    VADIterator = None

def simulate_realtime_processing(file_path, vad_model, whisper_model, german_dict, lm_tokenizer, lm_model):
    """
    Simulates real-time processing of an audio file to measure latency and RTF.
    """
    # Normalization constants from best model
    CONFIDENCE_MIN = -2.0
    CONFIDENCE_MAX = -0.1
    PERPLEXITY_MIN = 10.0
    PERPLEXITY_MAX = 10000.0

    # Final tuned Thresholds for GCS classification from best model
    threshold_params = {
        'conf_short': 0.6,
        'lex_gcs2': 0.5,
        'conf_gcs2': 0.45,
        'ppl_gcs45': 0.75,
        'conf_gcs45': 0.55
    }
    if VADIterator is None:
        print("VADIterator not available. Cannot run simulation.")
        return

    print(f"\n--- Starting REAL-TIME SIMULATION for: {os.path.basename(file_path)} ---")
    
    waveform, sampling_rate = load_audio(file_path)
    if waveform is None:
        return
    
    if sampling_rate != 16000:
        try:
            import librosa
            waveform = librosa.resample(y=waveform, orig_sr=sampling_rate, target_sr=16000)
            sampling_rate = 16000
            print("Audio resampled to 16kHz for simulation.")
        except ImportError:
            print("Librosa not installed. Cannot resample.")
            return

    # Apparently the VAD model strictly requires 512 sample chunks for 16kHz audio.
    window_size_samples = 512
    
    vad_iterator = VADIterator(vad_model, threshold=0.5, sampling_rate=sampling_rate)
    
    print(f"Simulating audio stream with {window_size_samples}-sample chunks...")
    all_metrics = []
    
    speech_start_time = None
    
    for i in range(0, len(waveform), window_size_samples):
        chunk = waveform[i : i + window_size_samples]
        if len(chunk) < window_size_samples:
            break 
            
        speech_dict = vad_iterator(chunk, return_seconds=True)
        
        if speech_dict:
            if 'start' in speech_dict:
                speech_start_time = speech_dict['start']
                print(f"Speech detected starting at ~{speech_start_time:.2f}s in stream.")
            
            if 'end' in speech_dict and speech_start_time is not None:
                speech_end_time = speech_dict['end']
                segment_end_time_in_stream = (i + window_size_samples) / sampling_rate
                print(f"Speech segment ended at ~{speech_end_time:.2f}s.")
                
                pipeline_start_time = time.perf_counter()
                
                start_sample = int(speech_start_time * sampling_rate)
                end_sample = int(speech_end_time * sampling_rate)
                speech_chunk_waveform = waveform[start_sample:end_sample]
                speech_duration_s = len(speech_chunk_waveform) / sampling_rate

                if speech_duration_s < 0.2: # Skip very short, likely false positive segments (fine-tune later)
                    print("  -> Segment too short, skipping.")
                    speech_start_time = None # Reset for next segment
                    continue

                print(f"  -> Processing segment from {speech_start_time:.2f}s to {speech_end_time:.2f}s...")
                
                transcription_result = transcribe_chunk(speech_chunk_waveform, whisper_model)
                if transcription_result and transcription_result['text']:
                    # A. Normalize features
                    norm_conf = normalize_feature(transcription_result['avg_logprob'], CONFIDENCE_MIN, CONFIDENCE_MAX)
                    norm_lex = np.clip(calculate_lexical_validity(transcription_result['text'], german_dict), 0.0, 1.0)
                    norm_inv_ppl = normalize_feature(calculate_perplexity(transcription_result['text'], lm_tokenizer, lm_model), PERPLEXITY_MIN, PERPLEXITY_MAX, invert=True)
                    
                    normalized_features = {
                        'norm_confidence': norm_conf,
                        'norm_lexical': norm_lex,
                        'norm_inv_perplexity': norm_inv_ppl
                    }
                    raw_features = {'transcription': transcription_result['text']}
                    no_speech_prob = transcription_result.get('no_speech_prob', 0)
                    
                    # B. Classify using the final rule-based function
                    gcs_level = classify_gcs_level_rule_based(normalized_features, no_speech_prob, raw_features, threshold_params)
                else:
                    gcs_level = 1 # No speech detected by Whisper or empty transcript
                
                pipeline_end_time = time.perf_counter()
                
                processing_time = pipeline_end_time - pipeline_start_time
                latency = (pipeline_end_time - pipeline_start_time) + (segment_end_time_in_stream - speech_end_time)
                rtf = processing_time / speech_duration_s if speech_duration_s > 0 else float('inf')
                
                print(f"    -> Processed in {processing_time:.2f}s. "
                      f"Latency: {latency:.2f}s. RTF: {rtf:.2f}. "
                      f"GCS Prediction: {gcs_level}")
                      
                all_metrics.append({'latency_s': latency, 'rtf': rtf, 'duration_s': speech_duration_s})
                
                speech_start_time = None
    
    vad_iterator.reset_states()
    
    if all_metrics:
        avg_latency = np.mean([m['latency_s'] for m in all_metrics])
        avg_rtf = np.mean([m['rtf'] for m in all_metrics])
        print("\n--- Simulation Complete ---")
        print(f"Average End-to-End Latency: {avg_latency:.3f} seconds")
        print(f"Average Real-Time Factor (RTF): {avg_rtf:.3f}")
        print("(RTF < 1.0 means faster than real-time)")
    else:
        print("\n--- Simulation Complete ---")
        print("No speech segments were processed to calculate average metrics.")

if __name__ == "__main__":
    # --- Configuration ---
    WHISPER_MODEL_SIZE = "base" # "tiny", "base", "small", "medium"
    LM_MODEL_ID = "dbmdz/german-gpt2"
    FEATURE_WEIGHTS = {'confidence': 0.3, 'lexical': 0.3, 'perplexity': 0.4}
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_audio_path = os.path.join(project_root, "data", "sample3.wav") 
    

    print("Initializing models for real-time simulation...")
    vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    whisper_model = initialize_whisper_model(model_size=WHISPER_MODEL_SIZE)
    german_dict = initialize_dictionary()
    lm_tokenizer, lm_model = initialize_language_model(model_id=LM_MODEL_ID)
    
    if all([vad_model, whisper_model, german_dict, lm_tokenizer, lm_model]):
        simulate_realtime_processing(input_audio_path, vad_model, whisper_model, german_dict,
                                     lm_tokenizer, lm_model)
    else:
        print("Model initialization failed. Exiting simulation.")