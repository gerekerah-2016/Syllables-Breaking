"""
Train tokenizer - FIXED to preserve all Ethiopic punctuation and prevent spaces in tags
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import os
import sentencepiece as spm
from src.logger import get_logger
from src.params import get_run_params
from src.utils.path_utils import get_logs_dir


def train_tokenizer(tokenizer_type, vocab_size, input_path, output_path):
    logger = get_logger()
    is_splintered = get_run_params("IS_ENCODED")
    
    logger.info(f"Training {tokenizer_type} tokenizer (vocab: {vocab_size})")
    logger.info(f"Mode: {'SPLINTER' if is_splintered else 'BASELINE'}")
    
    if is_splintered:
        # SPLINTERED mode - preserve all Ethiopic punctuation
        spm.SentencePieceTrainer.Train(
            input=input_path,
            model_prefix=output_path,
            vocab_size=vocab_size,
            model_type=tokenizer_type,
            character_coverage=1.0,#1.0
            byte_fallback=False,
            split_digits=False,
            split_by_unicode_script=False,
            split_by_whitespace=True,
            treat_whitespace_as_suffix=True,
            # ADD THESE FOR FASTER TRAINING
            #input_sentence_size=000000,   # Only use 1 million sentences
            #shuffle_input_sentence=True,    # Randomly select sentences
            # ADD THESE TO PREVENT SPACES IN TAGS
            allow_whitespace_only_pieces=False,
            remove_extra_whitespaces=False,
            # Include ALL Ethiopic punctuation as user-defined symbols
            user_defined_symbols=['፡', '።', '፣', '፤', '፥', '፦', '፧', '፠', '፨', '⟨', '⟩'],
            max_sentence_length=8192,#8192
            num_threads=4,#4
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
    else:
        # BASELINE mode
        spm.SentencePieceTrainer.Train(
            input=input_path,
            model_prefix=output_path,
            vocab_size=vocab_size,
            model_type=tokenizer_type,
            character_coverage=1.0,#1.0,
            byte_fallback=False,
            split_digits=True,
            split_by_unicode_script=True,
            split_by_whitespace=True,
            treat_whitespace_as_suffix=True,
            num_threads=4,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
    
    logger.info(f"Training complete: {output_path}")