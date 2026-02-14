import numpy as np

def normalize_feature(value, min_val, max_val, invert=False):
    """
    Normalizes a value to a 0-1 range.
    Clamps the value to be within min_val and max_val.
    """
    clamped_value = np.clip(value, min_val, max_val)

    normalized = (clamped_value - min_val) / (max_val - min_val)
    
    if invert:
        return 1.0 - normalized
    else:
        return normalized

def calculate_combined_comprehensibility_score(features, weights):
    """
    Calculates a combined score from normalized features using given weights.

    Args:
        features (dict): A dictionary with raw feature values, e.g.,
                         {'avg_logprob': -0.5, 'lexical_validity': 0.9, 'perplexity': 150.0}
        weights (dict): A dictionary with weights for each feature, e.g.,
                        {'confidence': 0.4, 'lexical': 0.3, 'perplexity': 0.3}

    Returns:
        float: A combined comprehensibility score between 0.0 and 1.0.
    """
    # --- Define Normalization Ranges (These are initial guesses based on observation!) ---
    # Need to tune these based on actual data later
    # For avg_logprob, less negative is better. A typical range  [-2.0, -0.1]
    CONFIDENCE_MIN = -2.0
    CONFIDENCE_MAX = -0.1
    
    # Perplexity, lower is better. Caped to handle extreme values.
    # A log scale might be better, have to test that later.
    PERPLEXITY_MIN = 10.0 # Anything below this is considered very good
    PERPLEXITY_MAX = 5000.0 # Cap perplexity for normalization
    
    # --- Normalize each feature to a 0-1 scale where 1.0 is "good" ---
    
    # Confidence: Higher is better
    norm_confidence = normalize_feature(
        features.get('avg_logprob', CONFIDENCE_MIN),
        CONFIDENCE_MIN,
        CONFIDENCE_MAX
    )
    
    # Lexical Validity
    norm_lexical = np.clip(features.get('lexical_validity', 0.0), 0.0, 1.0)
    
    # Perplexity: Lower is better
    norm_inv_perplexity = normalize_feature(
        features.get('perplexity', PERPLEXITY_MAX),
        PERPLEXITY_MIN,
        PERPLEXITY_MAX,
        invert=True
    )
    
    # --- Calculate weighted sum ---
    combined_score = (
        norm_confidence * weights.get('confidence', 0.33) +
        norm_lexical * weights.get('lexical', 0.33) +
        norm_inv_perplexity * weights.get('perplexity', 0.34)
    )
    
    return combined_score, {'norm_confidence': norm_confidence, 'norm_lexical': norm_lexical, 'norm_inv_perplexity': norm_inv_perplexity}

def classify_gcs_level_rule_based(normalized_features, no_speech_prob):
    """
    Classifies GCS level using a rule-based approach on normalized features.
    """
    if no_speech_prob > 0.8:
        return 1
    
    # Ensure transcription is a string before splitting
    transcription_text = str(raw_features.get('transcription', ''))
    num_words = len(transcription_text.split())

    # --- Rule 1: Handle very short utterances ---
    if num_words <= 2:
        if normalized_features['norm_confidence'] > thresholds.get['conf_short']:
            return 5
        else:
            return 2

    # --- Rule 2: Handle longer utterances ---
    if normalized_features['norm_lexical'] < thresholds.get['lex_gcs2'] or \
       normalized_features['norm_confidence'] < thresholds.get['conf_gcs2']:
        return 2

    if normalized_features['norm_inv_perplexity'] > thresholds.get['ppl_gcs45'] and \
       normalized_features['norm_confidence'] > thresholds.get['conf_gcs45']:
        return 5
        
    return 3

''' # --- Rule 1: The GCS 2 "Lexical" Gate ---
    # If the lexical validity is extremely low, it's almost certainly GCS 2.
    # We also check confidence to make sure it's not a confident whisper of a non-word.
    LEXICAL_THRESHOLD_FOR_GCS2 = 0.3
    CONFIDENCE_THRESHOLD_FOR_GCS2 = 0.4
    if normalized_features['norm_lexical'] < LEXICAL_THRESHOLD_FOR_GCS2 and \
       normalized_features['norm_confidence'] < CONFIDENCE_THRESHOLD_FOR_GCS2:
        return 2
        
    # --- Rule 2: The GCS 4/5 "Perplexity" Gate ---
    # If the perplexity is very low (meaning coherence is high), it's GCS 4/5.
    PERPLEXITY_THRESHOLD_FOR_GCS45 = 0.8
    if normalized_features['norm_inv_perplexity'] > PERPLEXITY_THRESHOLD_FOR_GCS45:
        return 5
        
    # --- Rule 3: The GCS 3 "Default" ---
    # If it passed the GCS 2 gate (it's made of words) but failed the
    # GCS 4/5 gate (it's not coherent), then it's GCS 3.
    return 3'''

# --- Test Block ---
if __name__ == "__main__":
    print("--- Testing classifier.py ---")

    # --- Step 1: Define Normalization Ranges and Rule Thresholds ---
    CONFIDENCE_MIN = -2.0
    CONFIDENCE_MAX = -0.1
    PERPLEXITY_MIN = 10.0
    PERPLEXITY_MAX = 5000.0

    def test_classifier_rule_based(normalized_features, no_speech_prob):
        # These thresholds are for testing the logic. You tune them in the notebook.
        LEXICAL_THRESHOLD_FOR_GCS2 = 0.3
        CONFIDENCE_THRESHOLD_FOR_GCS2 = 0.4
        PERPLEXITY_THRESHOLD_FOR_GCS45 = 0.8

        if no_speech_prob > 0.8: return 1
        if normalized_features['norm_lexical'] < LEXICAL_THRESHOLD_FOR_GCS2 and \
           normalized_features['norm_confidence'] < CONFIDENCE_THRESHOLD_FOR_GCS2:
            return 2
        if normalized_features['norm_inv_perplexity'] > PERPLEXITY_THRESHOLD_FOR_GCS45:
            return 5
        return 3

    # --- Step 2: Define simulated raw feature sets ---
    gcs_5_features = {'avg_logprob': -0.2, 'lexical_validity': 1.0, 'perplexity': 50.0, 'no_speech_prob': 0.01}
    gcs_3_features = {'avg_logprob': -0.8, 'lexical_validity': 1.0, 'perplexity': 10000.0, 'no_speech_prob': 0.1}
    gcs_2_features = {'avg_logprob': -2.5, 'lexical_validity': 0.1, 'perplexity': 20000.0, 'no_speech_prob': 0.4}
    
    test_cases = {
        "Simulated GCS 5 (Coherent)": gcs_5_features,
        "Simulated GCS 3 (Word Salad)": gcs_3_features,
        "Simulated GCS 2 (Incomprehensible)": gcs_2_features
    }

    # --- Step 3: Loop through test cases, normalize, and classify ---
    for name, features in test_cases.items():
        # A. Normalize the raw features
        norm_conf = normalize_feature(features['avg_logprob'], CONFIDENCE_MIN, CONFIDENCE_MAX)
        norm_lex = np.clip(features['lexical_validity'], 0.0, 1.0)
        norm_inv_ppl = normalize_feature(features['perplexity'], PERPLEXITY_MIN, PERPLEXITY_MAX, invert=True)
        
        normalized_features_dict = {
            'norm_confidence': norm_conf,
            'norm_lexical': norm_lex,
            'norm_inv_perplexity': norm_inv_ppl
        }

        # B. Classify using the rule-based function
        level = test_classifier_rule_based(normalized_features_dict, features['no_speech_prob'])
        
        # C. Print results
        print(f"\n{name}:")
        print(f"  Raw Features: {features}")
        print(f"  Normalized Features: { {k: f'{v:.2f}' for k, v in normalized_features_dict.items()} }")
        print(f"  -> Classified as GCS Level: {level}")