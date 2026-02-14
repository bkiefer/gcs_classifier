import string
import enchant 
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

def initialize_dictionary(lang_code="de_DE"):
    """
    Initializes the dictionary for a given language.

    Args:
        lang_code (str): The language code (e.g., "de_DE" for German).

    Returns:
        enchant.Dict: An initialized dictionary object, or None if error.
    """
    try:
        dictionary = enchant.Dict(lang_code)
        print(f"PyEnchant dictionary '{lang_code}' initialized successfully.")
        return dictionary
    except enchant.errors.DictNotFoundError:
        print(f"Error: Dictionary for '{lang_code}' not found.")
        return None

def calculate_lexical_validity(text, german_dictionary):
    """
    Calculates the ratio of valid German words in a given text.

    Args:
        text (str): The transcribed text.
        german_dictionary (enchant.Dict): The initialized PyEnchant dictionary.

    Returns:
        float: The ratio of valid words (0.0 to 1.0).
    """
    if not text or not german_dictionary:
        return 0.0

    # 1. Preprocess the text: remove punctuation and make lowercase
    # Creates a translation table that maps each punctuation character to None
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.lower().translate(translator)

    # 2. Split into words
    words = clean_text.split()
    if not words:
        return 0.0

    # 3. Check each word
    valid_word_count = 0
    for word in words:
        if german_dictionary.check(word):
            valid_word_count += 1
    
    # 4. Calculate ratio
    ratio = valid_word_count / len(words)
    return ratio

def initialize_language_model(model_id="dbmdz/german-gpt2"):
    """
    Initializes and loads a pre-trained German language model and tokenizer
    from Hugging Face.

    Args:
        model_id (str): The identifier for the Hugging Face model.
                        "dbmdz/german-gpt2" is a good German GPT-2 model.
                        "distilgpt2" is smaller/faster but not German-specific.

    Returns:
        tuple: A tuple containing:
            - tokenizer: The loaded tokenizer.
            - model: The loaded language model.
        Returns (None, None) if an error occurs.
    """
    try:
        print(f"Initializing Hugging Face model '{model_id}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = GPT2LMHeadModel.from_pretrained(model_id)
        print("Hugging Face model and tokenizer initialized successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error initializing Hugging Face model: {e}")
        return None, None

def calculate_perplexity(text, tokenizer, model):
    """
    Calculates the perplexity of a given text using a pre-trained LM.
    Lower perplexity indicates a more plausible sentence.

    Args:
        text (str): The transcribed text.
        tokenizer: The initialized Hugging Face tokenizer.
        model: The initialized Hugging Face language model.

    Returns:
        float: The perplexity score. Returns a very high value for empty/short text.
    """
    if not text or not tokenizer or not model:
        return float('inf') 

    try:
        inputs = tokenizer(text, return_tensors="pt")
    except Exception as e:
        print(f"Warning: Tokenizer failed for text '{text}'. Error: {e}")
        return float('inf')
        
    if inputs.input_ids.size(1) == 0:
        return float('inf')

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        neg_log_likelihood = outputs.loss

    perplexity = torch.exp(neg_log_likelihood)

    return perplexity.item()

# --- Test Block ---
if __name__ == "__main__":
    print("--- Testing feature_extractor.py (Lexical Validity) ---")
    
    # Initialize dictionary
    german_dict = initialize_dictionary()

    if german_dict:
        # Test cases
        test_case_1 = "Der schnelle braune Fuchs springt über den faulen Hund."
        test_case_2 = "Das ist ein perfekter Satz"
        test_case_3 = "Himmel viel Auto rennen Straße" # "Word Salad"
        test_case_4 = "asdfgh jkl qwertz" # Gibberish non-words
        test_case_5 = "" # Empty string

        validity_1 = calculate_lexical_validity(test_case_1, german_dict)
        validity_2 = calculate_lexical_validity(test_case_2, german_dict)
        validity_3 = calculate_lexical_validity(test_case_3, german_dict)
        validity_4 = calculate_lexical_validity(test_case_4, german_dict)
        validity_5 = calculate_lexical_validity(test_case_5, german_dict)

        # Not 100% accurate, but should be close to 1.0 for valid sentences
        # Maybe inflected forms are problematic
        # Could improve by using a more advanced NLP library like spaCy or NLTK
        # But for now, this should be sufficient for basic validation
        # Could influence the perfomance otherwise
        print(f"'{test_case_1}' -> Lexical Validity: {validity_1:.2f}")
        print(f"'{test_case_2}' -> Lexical Validity: {validity_2:.2f}")
        print(f"'{test_case_3}' -> Lexical Validity: {validity_3:.2f}") # Should be 1.0
        print(f"'{test_case_4}' -> Lexical Validity: {validity_4:.2f}") # Should be 0.0
        print(f"'{test_case_5}' -> Lexical Validity: {validity_5:.2f}")

    # --- LM Coherence (Perplexity) Test ---
    print("\n--- Testing LM Coherence (Perplexity) ---")
    lm_tokenizer, lm_model = initialize_language_model()

    if lm_tokenizer and lm_model:

        # Should have low perplexity.
        ppl_1 = calculate_perplexity(test_case_1, lm_tokenizer, lm_model)
        
        # "Word salad" with real words should have very high perplexity.
        ppl_3 = calculate_perplexity(test_case_3, lm_tokenizer, lm_model)
        
        # Gibberish non-words will also have very high perplexity.
        ppl_4 = calculate_perplexity(test_case_4, lm_tokenizer, lm_model)
        
        test_case_good = "Ich brauche dringend medizinische Hilfe."
        ppl_good = calculate_perplexity(test_case_good, lm_tokenizer, lm_model)

        print(f"'{test_case_1}' -> Perplexity: {ppl_1:,.2f}")
        print(f"'{test_case_good}' -> Perplexity: {ppl_good:,.2f}")
        print(f"'{test_case_3}' -> Perplexity: {ppl_3:,.2f}  <-- Should be much higher")
        print(f"'{test_case_4}' -> Perplexity: {ppl_4:,.2f}  <-- Should be very high")