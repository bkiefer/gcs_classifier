
# Automated Speech Comprehensibility Assessment for GCS

**Author:** Marlow Springmeier

The function comments were created with the help of Git Copilot

Contains the complete source code, data preparation scripts and experimental notebooks for the bachelor thesis titled "Automatically Evaluating Speech Comprehensibility for Use with the Glasgow Coma Scale."

The project is structured to be reproducible, with dedicated scripts for data preparation, feature extraction, and evaluation.

## Repository Structure

The project is organized into the following key directories:

```
GCS_Speech_Thesis/
├── data/              # Houses all audio data, split into validation and test sets.
├── notebooks/         # Jupyter Notebooks for analysis, tuning, and evaluation.
├── results/           # Output directory for CSV feature sets and plots.
├── scripts/           # Standalone Python scripts for data preparation and batch processing.
├── src/               # Core Python source code for the pipeline modules.
├── .gitignore         # Files and folders for Git to ignore.
└── requirements.txt   # Lists all Python library dependencies.
```

## Setup and Installation

This project was developed on Linux Mint 21.x using Python 3.12. 


**Install Dependencies:**
All required Python libraries are listed in `requirements.txt`. Install them using pip:
```bash
pip install -r requirements.txt
```

**Install System Dependencies:**
This project relies on two system-level packages:
*   **Hunspell German Dictionary (for `pyenchant`):**
    ```bash
    sudo apt-get install hunspell-de-de
    ```
*   **FFmpeg (for `gTTS` audio conversion):**
    ```bash
    sudo apt-get install ffmpeg
    ```

## How to Run the Project: A Step-by-Step Workflow

The project is designed to be run in a sequential order, from data preparation to final evaluation.

### Step 1: Data Preparation

*(Note: Users must provide their own source audio files in the locations specified within the scripts.)*

1.  **Prepare Source Data:** Place source audio files in the appropriate folders (e.g., `data/CommonVoice21.0/...`, `data/noise_samples/`, `data/gcs_level_2_incomprehensible/`).
2.  **Generate Word Salad (GCS 3 Data):** Run the `gTTS` script to create the synthetic "word salad" audio files.
    ```bash
    python scripts/generate_word_salad.py
    ```
3.  **Split Datasets:** Run the preparation script to randomly split all source data into `validation_set` and `test_set` directories.
    ```bash
    python scripts/prepare_datasets.py
    ```
4.  **Augment with Noise:** Generate the noisy versions of the GCS 4/5 data for both the validation and test sets.
    ```bash
    python scripts/augment_noise.py
    ```

### Step 2: Batch Feature Extraction

This is the most computationally intensive step. The script processes all audio files from the prepared validation and test sets and creates two master CSV files in the `results/` directory.

```bash
python scripts/run_batch_processing.py
```

### Step 3: Analysis and Model Training (Jupyter Notebooks)

All analysis and model training is performed in Jupyter Notebooks located in the `notebooks/` directory. 

```bash
jupyter lab
```

1.  **`RQ1_Noise_Analysis.ipynb`**: Loads the `features_validation_set.csv` and generates the plots for analyzing feature stability under noise.
2.  **`ML_Classifier_Training.ipynb`**: This is the main notebook for classifier development. It loads the validation data, visualizes feature effectiveness (RQ2), and trains and evaluates the final Decision Tree classifier (answering RQ3 & RQ4 for the validation set).
3.  **`Final_Evaluation_ML_Model.ipynb`**: This notebook loads the **unseen `features_test_set.csv`** and applies the final, trained classifier to generate the definitive performance metrics reported in the thesis.

### Step 4: Real-Time Performance Benchmark (RQ5)

The `standalone_realtime_test.py` script is used to measure latency and RTF. The model size and input file can be configured within the script's `__main__` block.

```bash
python src/standalone_realtime_test.py
```

## Core Modules (`src/` directory)

*   **`audio_utils.py`**: Contains functions for loading audio and the two-stage VAD logic for robust speech segmentation.
*   **`transcription.py`**: Handles the initialization of the FasterWhisper model and the transcription of individual audio chunks.
*   **`feature_extractor.py`**: Implements the calculation of the three core linguistic features: lexical validity and language model perplexity.
*   **`classifier.py`**: Contains the logic for feature normalization and the final `classify_gcs_level_rule_based` function.

