"""
Path utilities for experiment management.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import os
from pathlib import Path
from src.params import get_run_params  # This is the ONLY import needed


def create_experiment_dirs():
    """Create all experiment directories."""
    base_dir = Path(get_run_params("OUTPUT_BASE_DIR"))
    experiment_name = get_run_params("EXPERIMENT_NAME")
    
    exp_dir = base_dir / experiment_name
    
    # Create subdirectories
    dirs = [
        exp_dir,
        exp_dir / "tokenizers",
        exp_dir / "corpora",
        exp_dir / "tokenized_corpora",
        exp_dir / "decoded_corpora",  # NEW: directory for decoded files
        exp_dir / "static_checks",
        exp_dir / "logs",
        exp_dir / "data" / "word_dict",
        exp_dir / "results" / "splinter",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def get_experiment_dir() -> Path:
    """Get the current experiment directory."""
    return Path(get_run_params("OUTPUT_BASE_DIR")) / get_run_params("EXPERIMENT_NAME")


def get_logs_dir() -> Path:
    """Get directory for logs."""
    exp_dir = get_experiment_dir()
    task_id = get_run_params("TASK_ID")
    timestamp = get_run_params("TIMESTAMP")
    return exp_dir / "logs" / f"log-{task_id}-{timestamp}"


def get_raw_data_dir() -> Path:
    """Get directory for raw data (cached datasets)."""
    return Path("./raw_data")


def get_words_dict_dir() -> Path:
    """Get directory for word frequency dictionaries."""
    return get_experiment_dir() / "data" / "word_dict"


def get_splinter_dir() -> Path:
    """Get directory for splinter reduction maps."""
    return get_experiment_dir() / "results" / "splinter"


def get_tokenizer_path(tokenizer_type: str, vocab_size: int) -> Path:
    """Get path for a tokenizer model."""
    return get_experiment_dir() / "tokenizers" / f"{get_run_params('LANGUAGE')}_{tokenizer_type}_{vocab_size}"


def get_corpus_path(corpus_name: str) -> Path:
    """Get path for a corpus file."""
    return get_experiment_dir() / "corpora" / f"{corpus_name}_corpus.txt"


def get_tokenized_corpus_path(corpus_name: str, tokenizer_type: str, vocab_size: int) -> Path:
    """Get path for a tokenized corpus file."""
    return get_experiment_dir() / "tokenized_corpora" / f"{corpus_name}_{tokenizer_type}_{vocab_size}_tokenized.txt"


def get_decoded_corpus_path(corpus_name: str, tokenizer_type: str, vocab_size: int) -> Path:
    """Get path for a decoded corpus file."""
    return get_experiment_dir() / "decoded_corpora" / f"{corpus_name}_{tokenizer_type}_{vocab_size}_decoded.txt"


def get_static_checks_path() -> Path:
    """Get path for static checks results."""
    return get_experiment_dir() / "static_checks" / "results.json"


def get_corpus_name(corpus_path: str, dataset_name: str = None) -> str:
    """Generate corpus name from path."""
    if dataset_name:
        return dataset_name.replace('/', '_')
    return Path(corpus_path).stem


def delete_corpus_file_if_exists(corpus_name: str):
    """Delete corpus file if it exists."""
    corpus_path = get_corpus_path(corpus_name)
    if os.path.exists(corpus_path):
        os.remove(corpus_path)