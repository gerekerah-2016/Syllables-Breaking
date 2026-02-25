"""
Experiment parameters and configuration.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import copy
from datetime import datetime, timezone

# Global run parameters
_run_params = {}

def set_run_params(params):
    """Set global run parameters."""
    global _run_params
    _run_params = params
    _run_params["TIMESTAMP"] = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def get_run_params(key):
    """Get a run parameter by key."""
    global _run_params
    return _run_params[key]


def get_all_run_params():
    """Get all run parameters."""
    global _run_params
    return _run_params


# Configuration
LANGUAGE = 'ge'  # 'ge' for Ge'ez, 'am' for Amharic, 'ti' for Tigrinya
LANGUAGE_FULL = 'geez'
EXPERIMENT_DATE = '2026-02-17'

# Corpora for static checks
STATIC_CHECKS_CORPORA = ['original', 'splintered']

# ============================================================
# PAPER'S VOCABULARY SIZES (800 to 128K)
# ============================================================
PAPER_VOCAB_SIZES =  [4000, 6000,8000, 10000, 15000, 20000, 25000]

# Base experiment template with ALL parameters
experiment_template = {
    # Language settings
    'LANGUAGE': LANGUAGE,
    'LANGUAGE_FULL': LANGUAGE_FULL,
    
    # Output settings
    'OUTPUT_BASE_DIR': './outputs',
    'EXPERIMENT_NAME': f'{EXPERIMENT_DATE}-{LANGUAGE_FULL}-template',
    
    # Processing settings
    'IS_ENCODED': True,  # True for splintered, False for baseline
    'SAVE_CORPORA_INTO_FILE': True,  # Create corpora files
    'TRAIN_TOKENIZERS': True,  # Train tokenizers
    'TOKENIZE_CORPORA': True,  # Tokenize corpora
    'RUN_STATIC_CHECKS': True,  # Run static checks
    
    # Corpus settings - CRITICAL: Use the CLEAN data
    'SPLINTER_TRAINING_CORPUS_PATH': './Geez-Dataset-Clean',  # Use cleaned data
    'SPLINTER_TRAINING_CORPUS_NAME': 'geez_corpus',
    'SPLINTER_LETTERS_SUBSET': None,  # None = all letters
    
    # Hugging Face settings (for compatibility)
    'USE_HUGGINGFACE': False,
    'IS_HUGGINGFACE': False,
    'HF_DATASET_NAME': None,
    'HF_SPLIT': 'train',
    'HF_TEXT_FIELD': 'text',
    
    # Tokenizer settings
    'TOKENIZERS_TYPES': ['unigram', 'bpe'],
    
    # ============================================================
    # DEFAULT VOCABULARY SIZES (will be overridden by split function)
    # ============================================================
    'TOKENIZERS_VOCAB_SIZES': PAPER_VOCAB_SIZES,
    
    # Evaluation settings
    'STATIC_CHECKS_CORPORA': STATIC_CHECKS_CORPORA,
    
    # Task ID (set by main)
    'TASK_ID': 0,
    'TIMESTAMP': None,  # Set by set_run_params
}


def get_all_letters_template():
    """Template for experiment with all letters."""
    template = copy.deepcopy(experiment_template)
    template['EXPERIMENT_NAME'] = f'{EXPERIMENT_DATE}-{LANGUAGE_FULL}-all_letters'
    template['IS_ENCODED'] = True
    template['SPLINTER_LETTERS_SUBSET'] = None  # All consonants
    return template


def get_letters_subset_template():
    """Template for experiment with subset of letters."""
    template = copy.deepcopy(experiment_template)
    template['EXPERIMENT_NAME'] = f'{EXPERIMENT_DATE}-{LANGUAGE_FULL}-letters_subset'
    template['IS_ENCODED'] = True
    # Common consonants subset for testing
    template['SPLINTER_LETTERS_SUBSET'] = ["ሀ", "ለ", "ሐ", "መ", "ሠ", "ረ", "ሰ", "ሸ", "ቀ", "በ"]
    return template


def get_baseline_template():
    """Template for baseline experiment (no splintering)."""
    template = copy.deepcopy(experiment_template)
    template['EXPERIMENT_NAME'] = f'{EXPERIMENT_DATE}-{LANGUAGE_FULL}-baseline'
    template['IS_ENCODED'] = False
    template['SPLINTER_LETTERS_SUBSET'] = None
    return template


# ============================================================
# UPDATED: Single run with ALL vocabulary sizes (PROFESSIONAL)
# ============================================================
def split_to_separate_runs(template):
    """
    Create a SINGLE experiment run with ALL vocabulary sizes.
    This is the professional approach - one experiment per type
    that processes all vocabulary sizes.
    
    Args:
        template: Experiment template
        
    Returns:
        List with ONE experiment configuration containing all vocab sizes
    """
    runs = []
    run = copy.deepcopy(template)
    
    # Set ALL vocabulary sizes in ONE experiment
    run['TOKENIZERS_VOCAB_SIZES'] = PAPER_VOCAB_SIZES.copy()  # [800, 1000, 2000, 10000, 32000, 64000, 128000]
    
    runs.append(run)
    return runs


def get_dummy_experiment(experiment_name: str):
    """
    Get a dummy experiment for testing/demo.
    
    Args:
        experiment_name: Name for the experiment
        
    Returns:
        Experiment configuration
    """
    experiment = copy.deepcopy(get_baseline_template())
    experiment['EXPERIMENT_NAME'] = experiment_name
    experiment["TASK_ID"] = '1000000'
    experiment['TOKENIZERS_VOCAB_SIZES'] = [300]  # Just one small size for testing
    experiment['SAVE_CORPORA_INTO_FILE'] = False
    experiment['TRAIN_TOKENIZERS'] = False
    experiment['TOKENIZE_CORPORA'] = False
    return experiment


# ============================================================
# Create experiments list - now only 3 experiments total!
# ============================================================
experiments = []

# Add ALL letters experiment (splintered with all consonants)
experiments.extend(split_to_separate_runs(get_all_letters_template()))

# Add letters subset experiment (splintered with subset)
experiments.extend(split_to_separate_runs(get_letters_subset_template()))

# Add baseline experiment (no splintering)
experiments.extend(split_to_separate_runs(get_baseline_template()))

print(f"Created {len(experiments)} experiments")
print(f"Each experiment will process vocabulary sizes: {PAPER_VOCAB_SIZES}")