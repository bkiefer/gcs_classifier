from faster_whisper import WhisperModel
import time 


def initialize_whisper_model(model_size="base", device="cpu", compute_type="int8"):
    """
    Initializes and loads the FasterWhisper model.

    Args:
        model_size (str): The size of the Whisper model to load (e.g., "tiny", "base", "small").
        device (str): The device to run the model on ("cpu" or "cuda").
        compute_type (str): The quantization type to use (e.g., "int8", "float16").

    Returns:
        WhisperModel: The loaded FasterWhisper model object.
    """
    try:
        print(f"Initializing FasterWhisper model '{model_size}'...")
        # Downloads the model from Hugging Face on the first run and caches it.
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("FasterWhisper model initialized successfully.")
        return model
    except Exception as e:
        print(f"Error initializing Whisper model: {e}")
        return None

def transcribe_chunk(audio_chunk_waveform, whisper_model, language="de"):
    """
    Transcribes a single audio chunk using the provided FasterWhisper model.

    Args:
        audio_chunk_waveform (numpy.ndarray): The audio data for one segment.
        whisper_model (WhisperModel): The initialized FasterWhisper model.
        language (str): The language of the speech (e.g., "de" for German).

    Returns:
        dict: A dictionary containing transcription results for the chunk, or None if error.
    """
    if whisper_model is None:
        print("Whisper model not initialized.")
        return None

    try:
        start_time = time.perf_counter()
        
        # The transcribe function returns an iterator for segments and an info object.
        # For a short chunk, we usually expect only one segment.
        segments, info = whisper_model.transcribe(audio_chunk_waveform, language=language)
        
        segment_results = list(segments)
        
        end_time = time.perf_counter()
        transcription_time = end_time - start_time
        
        if not segment_results:
            # Possible if Whisper determines there is no speech in the chunk (despite VAD detecting it)
            print("Whisper detected no speech in this chunk.")
            return {
                'text': '',
                'avg_logprob': -99.0, # We use a very low logprob to indicate no speech
                'no_speech_prob': 1.0, # High probability of no speech
                'transcription_time_s': transcription_time
            }

        # For a single chunk, we'll just process the first detected segment.
        first_segment = segment_results[0]
        
        result_dict = {
            'text': first_segment.text.strip(),
            'avg_logprob': first_segment.avg_logprob,
            'no_speech_prob': first_segment.no_speech_prob,
            'transcription_time_s': transcription_time
        }
        
        print(f"Transcription complete in {transcription_time:.2f}s. Text: '{result_dict['text']}'")
        return result_dict

    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

# --- Test Block ---
if __name__ == "__main__":
    import os
    import numpy as np
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from src.audio_utils import load_audio, initialize_vad_model, get_speech_timestamps

    print("--- Testing transcription.py ---")

    # 1. Initialize the Whisper model
    # Change to "base" or "small" 
    whisper_model_instance = initialize_whisper_model(model_size="base")

    if whisper_model_instance:
        # 2. Load the same sample audio file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sample_audio_path = os.path.join(project_root, "data/CommonVoice21.0/cv-corpus-21.0-2025-03-14/de/clips", "TestSample.mp3")

        if not os.path.exists(sample_audio_path):
             print(f"Please place an audio file at {sample_audio_path} for testing.")
        else:
            waveform_data, sr = load_audio(sample_audio_path)

            if waveform_data is not None:
                # 3. Use VAD to find the first speech chunk to test
                vad_model_instance, vad_utils = initialize_vad_model()
                if vad_model_instance:
                    timestamps = get_speech_timestamps(waveform_data, vad_model_instance, vad_utils, sr)
                    if timestamps:
                        # Transcribe the first detected segment (only one for testing)
                        first_segment_ts = timestamps[0]
                        
                        # Convert times to sample indices
                        start_sample = int(first_segment_ts['start'] * sr)
                        end_sample = int(first_segment_ts['end'] * sr)
                        
                        # Slice the waveform to get the audio chunk
                        audio_chunk = waveform_data[start_sample:end_sample]

                        print(f"\nTranscribing first chunk ({first_segment_ts['start']:.2f}s to {first_segment_ts['end']:.2f}s)...")
                        # 4. Transcribe the chunk
                        transcription_result = transcribe_chunk(audio_chunk, whisper_model_instance, language="de")

                        if transcription_result:
                            print("\n--- Test Transcription Result ---")
                            print(f"  Text: {transcription_result['text']}")
                            print(f"  Avg Log Probability: {transcription_result['avg_logprob']:.4f}")
                            print(f"  No Speech Probability: {transcription_result['no_speech_prob']:.4f}")
                            print(f"  Transcription Time: {transcription_result['transcription_time_s']:.2f}s")
                    else:
                        print("VAD found no speech, cannot test transcription.")
    else:
        print("Failed to initialize Whisper model.")