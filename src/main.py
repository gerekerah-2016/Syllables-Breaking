"""
Main entry point for Ethiopic Syllable Breaking Pipeline.
Author: Gebreslassie Teklu Reda - PURE CJK VERSION
Date: 2026
"""

import os
import sys
import re
from pathlib import Path

# Set Hugging Face cache (optional)
os.environ['HF_HOME'] = 'D:/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = 'D:/huggingface_cache/datasets'

# Get the absolute path of the current file
current_file = os.path.abspath(__file__)
# Get the src directory (where this file is)
src_dir = os.path.dirname(current_file)
# Get the project root directory (parent of src)
project_root = os.path.dirname(src_dir)

# Add both to path
sys.path.insert(0, project_root)  # So we can import src.*
sys.path.insert(0, src_dir)        # So we can import directly

# Now we can import from src with correct paths
try:
    # Try importing with src prefix (when running from project root)
    from src.corpus.CorpusTokenizer import CorpusTokenizer
    from src.SplinterTrainer import SplinterTrainer
    from src.TextProcessorBaseline import TextProcessorBaseline
    from src.TextProcessorWithEncoding import TextProcessorWithEncoding
    from src.language_utils.LanguageUtilsFactory import LanguageUtilsFactory
    from src.logger import initialize_logger, get_logger
    from src.params import experiments, get_run_params, set_run_params
    from src.save_dataset_as_text_file import save_corpus_as_text_file
    from src.train_tokenizer import train_tokenizer
    
    # ============================================================
    # IMPORT UPDATED STATIC CHECKS WITH ALL EVALUATIONS
    # ============================================================
    from src.static_checks import (
        run_static_checks, 
        run_types_length_distribution,
        run_comprehensive_evaluation
    )
    
    # ============================================================
    # IMPORT ALL PATH UTILITIES
    # ============================================================
    from src.utils.path_utils import (
        create_experiment_dirs, 
        get_experiment_dir,
        get_tokenizer_path, 
        get_corpus_path, 
        get_tokenized_corpus_path,
        get_splinter_dir
    )
    
    from src.utils.utils import add_static_result_to_file, get_corpus_name
    
    # ============================================================
    # Import for decoding
    # ============================================================
    from src.decode_utils import SplinterDecoder
    
    # ============================================================
    # IMPORT DOWNSTREAM TASKS (NEW)
    # ============================================================
    from src.downstream.run_all import run_all_downstream_tasks
    
except ImportError as e:
    print(f"First import attempt failed: {e}")
    try:
        # If that fails, try importing without src prefix (when running from src dir)
        from corpus.CorpusTokenizer import CorpusTokenizer
        from SplinterTrainer import SplinterTrainer
        from TextProcessorBaseline import TextProcessorBaseline
        from TextProcessorWithEncoding import TextProcessorWithEncoding
        from language_utils.LanguageUtilsFactory import LanguageUtilsFactory
        from logger import initialize_logger, get_logger
        from params import experiments, get_run_params, set_run_params
        from save_dataset_as_text_file import save_corpus_as_text_file
        from train_tokenizer import train_tokenizer
        
        # ============================================================
        # IMPORT UPDATED STATIC CHECKS WITH ALL EVALUATIONS
        # ============================================================
        from static_checks import (
            run_static_checks, 
            run_types_length_distribution,
            run_comprehensive_evaluation
        )
        
        # ============================================================
        # IMPORT ALL PATH UTILITIES
        # ============================================================
        from utils.path_utils import (
            create_experiment_dirs, 
            get_experiment_dir,
            get_tokenizer_path, 
            get_corpus_path, 
            get_tokenized_corpus_path,
            get_splinter_dir
        )
        
        from utils.utils import add_static_result_to_file, get_corpus_name
        
        # ============================================================
        # Import for decoding
        # ============================================================
        from decode_utils import SplinterDecoder
        
        # ============================================================
        # IMPORT DOWNSTREAM TASKS (NEW)
        # ============================================================
        from downstream.run_all import run_all_downstream_tasks
        
    except ImportError as e2:
        print(f"Second import attempt failed: {e2}")
        print(f"Current sys.path: {sys.path}")
        print(f"Files in src directory: {os.listdir(src_dir)}")
        print(f"Files in src/corpus directory: {os.listdir(os.path.join(src_dir, 'corpus'))}")
        raise


# ============================================================
# SIMPLIFIED CLEANUP FUNCTION - Pure CJK version
# ============================================================
def final_cleanup(decoded_text):
    """
    Clean up decoded text - simplified for pure CJK approach.
    No brackets, no tags - just CJK characters.
    """
    if not decoded_text:
        return ""
    
    text = decoded_text

    # STEP 1: Remove SentencePiece boundary markers
    text = text.replace('▁', '')

    # STEP 2: Fix spacing between CJK characters if they were tokenized separately
    cjk_range = r'[\u4e00-\u9fff]'
    text = re.sub(f'({cjk_range})\\s+(?={cjk_range})', r'\1', text)
    
    # STEP 3: Fix Ethiopic punctuation spacing - remove space BEFORE, preserve after
    text = re.sub(r'\s+([፡።፣፤፥፦፧፠፨])', r'\1', text)
    
    # STEP 4: Fix specific patterns with numbers and parentheses
    text = re.sub(r'(\d+)\s*\(\s*:', r'\1(:', text)
    text = re.sub(r'\(\s*:', r'(:', text)
    text = re.sub(r':\s*\)', r':)', text)
    
    # STEP 5: Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def run():
    """Main pipeline execution."""
    
    # 1. Initialize language utils
    language_utils = LanguageUtilsFactory.get_by_language(get_run_params("LANGUAGE"))
    train_dataset_path = get_run_params("SPLINTER_TRAINING_CORPUS_PATH")
    train_dataset_name = get_run_params("SPLINTER_TRAINING_CORPUS_NAME")
    letters_subset = get_run_params("SPLINTER_LETTERS_SUBSET")
    
    # Initialize variables that might be set in splintered path
    reductions_map = None
    new_unicode_chars_map = None
    new_unicode_chars_inverted_map = None
    text_processor = None
    
    # 2. Handle Splintering vs. Baseline
    if get_run_params("SAVE_CORPORA_INTO_FILE"):
        
        if get_run_params("IS_ENCODED"):
            get_logger().info("Starting Splinter Training for Ethiopic...")
            splinter_trainer = SplinterTrainer(language_utils)
            
            # Train splinter rules
            reductions_map, new_unicode_chars_map, new_unicode_chars_inverted_map = splinter_trainer.train(
                train_dataset_path, train_dataset_name, letters_subset
            )
            
            # Create encoded processor
            text_processor = TextProcessorWithEncoding(
                language_utils,
                reductions_map,
                new_unicode_chars_map,
                new_unicode_chars_inverted_map
            )

        else:
            get_logger().info("Starting Baseline Training (No Splintering)...")
            text_processor = TextProcessorBaseline(language_utils)
        
        # 3. Save processed corpus
        save_corpus_as_text_file(text_processor, train_dataset_path, train_dataset_name)
        language_utils.save_additional_corpora_for_evaluation(text_processor)
    
    # 4. Train tokenizers
    if get_run_params("TRAIN_TOKENIZERS"):
        corpus_name = "splintered" if get_run_params("IS_ENCODED") else "baseline"
        tokenizer_corpus_path = get_corpus_path(corpus_name)
        
        for tokenizer_type in get_run_params("TOKENIZERS_TYPES"):
            for vocab_size in get_run_params("TOKENIZERS_VOCAB_SIZES"):
                tokenizer_path = get_tokenizer_path(tokenizer_type=tokenizer_type, vocab_size=vocab_size)
                train_tokenizer(
                    tokenizer_type=tokenizer_type,
                    vocab_size=vocab_size,
                    input_path=tokenizer_corpus_path,
                    output_path=tokenizer_path
                )
    
    # 5. Tokenize corpora and run static checks
    for tokenizer_type in get_run_params("TOKENIZERS_TYPES"):
        for vocab_size in get_run_params("TOKENIZERS_VOCAB_SIZES"):
            
            if get_run_params("TOKENIZE_CORPORA"):
                tokenizer_path = get_tokenizer_path(tokenizer_type=tokenizer_type, vocab_size=vocab_size)
                
                for corpus_name in get_run_params("STATIC_CHECKS_CORPORA"):
                    corpus_path = get_corpus_path(corpus_name)
                    tokenized_corpus_path = get_tokenized_corpus_path(corpus_name, tokenizer_type, vocab_size)
                    
                    # Tokenize corpus
                    CorpusTokenizer().tokenize_corpus_into_file(
                        tokenizer_path, corpus_path, tokenized_corpus_path
                    )
            
            if get_run_params("RUN_STATIC_CHECKS"):
                get_logger().info(f"Running evaluation for {tokenizer_type} v{vocab_size}")
                
                # Type-length distribution
                types_length_distribution = run_types_length_distribution(tokenizer_type, vocab_size)
                add_static_result_to_file(types_length_distribution)
                
                # Static checks for each corpus
                for corpus_name in get_run_params("STATIC_CHECKS_CORPORA"):
                    corpus_path = get_corpus_path(corpus_name)
                    tokenized_corpus_path = get_tokenized_corpus_path(corpus_name, tokenizer_type, vocab_size)
                    tokenizer_path = get_tokenizer_path(tokenizer_type=tokenizer_type, vocab_size=vocab_size)
                    
                    results = run_static_checks(corpus_path, tokenized_corpus_path, tokenizer_path, vocab_size)
                    add_static_result_to_file(results)
    
    # ============================================================
    # Decode tokenized corpora (only for SPLINTERED experiments)
    # ============================================================
    if get_run_params("IS_ENCODED") and get_run_params("TOKENIZE_CORPORA"):
        get_logger().info("=" * 60)
        get_logger().info("DECODING TOKENIZED CORPORA")
        get_logger().info("=" * 60)
        
        decoder = SplinterDecoder(language_utils)
        if decoder.load_decode_map():
            for tokenizer_type in get_run_params("TOKENIZERS_TYPES"):
                for vocab_size in get_run_params("TOKENIZERS_VOCAB_SIZES"):
                    for corpus_name in get_run_params("STATIC_CHECKS_CORPORA"):
                        if "splintered" in corpus_name:
                            tokenized_path = get_tokenized_corpus_path(corpus_name, tokenizer_type, vocab_size)
                            
                            # Create output path for final cleaned version
                            exp_dir = get_experiment_dir()
                            decoded_dir = exp_dir / "decoded_corpora"
                            decoded_dir.mkdir(exist_ok=True)
                            
                            # ============================================================
                            # Decode AND clean in one step - only final output
                            # ============================================================
                            final_path = decoded_dir / f"{corpus_name}_{tokenizer_type}_{vocab_size}.txt"
                            
                            if tokenized_path.exists():
                                # Decode line by line and apply cleanup immediately
                                with open(tokenized_path, 'r', encoding='utf-8') as f_in:
                                    with open(final_path, 'w', encoding='utf-8') as f_out:
                                        for line in f_in:
                                            # Decode the line
                                            decoded_line = decoder.decode_line(line.strip())
                                            # Apply final cleanup
                                            cleaned_line = final_cleanup(decoded_line)
                                            f_out.write(cleaned_line + '\n')
                                
                                get_logger().info(f"✅ Final decoded output saved to {final_path}")
                                
                            else:
                                get_logger().warning(f"Tokenized file not found: {tokenized_path}")
            
            get_logger().info("✓ Decoding complete - check the 'decoded_corpora' folder")
        else:
            get_logger().error("❌ Failed to load decode map - decoding skipped")
    
    # ============================================================
    # RUN COMPREHENSIVE EVALUATION (only for splintered experiments)
    # ============================================================
    if get_run_params("IS_ENCODED") and get_run_params("RUN_STATIC_CHECKS"):
        get_logger().info("=" * 60)
        get_logger().info("RUNNING COMPREHENSIVE EVALUATION")
        get_logger().info("=" * 60)
        get_logger().info("This will compute:")
        get_logger().info("  • Vocabulary overlap (Figure 2)")
        get_logger().info("  • Rényi efficiency (Tables 2-4)")
        get_logger().info("  • Distinct neighbors (Figure 3)")
        get_logger().info("  • Token distribution statistics")
        get_logger().info("=" * 60)
        
        try:
            # Run all evaluations with maps for PhD components
            results = run_comprehensive_evaluation(
                encode_map=new_unicode_chars_map,
                decode_map=new_unicode_chars_inverted_map,
                language_utils=language_utils
            )
            
            get_logger().info("✓ Comprehensive evaluation complete")
            get_logger().info(f"  Results saved to: {get_experiment_dir() / 'static_checks' / 'comprehensive_results.json'}")
            
        except Exception as e:
            get_logger().error(f"Comprehensive evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ============================================================
    # RUN DOWNSTREAM TASKS (only for splintered experiments)
    # ============================================================
    if get_run_params("IS_ENCODED") and text_processor is not None:
        get_logger().info("=" * 60)
        get_logger().info("RUNNING DOWNSTREAM TASKS")
        get_logger().info("=" * 60)
        get_logger().info("This will run:")
        get_logger().info("  • Named Entity Recognition (MasakhaNER)")
        get_logger().info("  • Part-of-Speech Tagging (UD Amharic)")
        get_logger().info("  • Machine Translation (JW300)")
        get_logger().info("  • Text Classification (Amharic News)")
        get_logger().info("=" * 60)
        
        try:
            exp_dir = get_experiment_dir()
            downstream_results = run_all_downstream_tasks(text_processor, exp_dir)
            
            get_logger().info("✓ Downstream tasks complete")
            get_logger().info(f"  Results saved to: {exp_dir / 'static_checks' / 'downstream_results.json'}")
            
        except ImportError as e:
            get_logger().error(f"Downstream tasks module not found: {e}")
            get_logger().info("To run downstream tasks, ensure src/downstream/ exists with all required files")
            get_logger().info("Required files:")
            get_logger().info("  • src/downstream/__init__.py")
            get_logger().info("  • src/downstream/ner.py")
            get_logger().info("  • src/downstream/pos.py")
            get_logger().info("  • src/downstream/mt.py")
            get_logger().info("  • src/downstream/classification.py")
            get_logger().info("  • src/downstream/run_all.py")
            
        except Exception as e:
            get_logger().error(f"Downstream tasks failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        if not get_run_params("IS_ENCODED"):
            get_logger().info("⏭️ Skipping downstream tasks (baseline experiment)")
        elif text_processor is None:
            get_logger().warning("⚠️ Text processor not available - skipping downstream tasks")


if __name__ == '__main__':
    # Support for parallel runs
    slurm_array_task_id = os.getenv('SLURM_ARRAY_TASK_ID')
    task_id = (int(slurm_array_task_id) - 1) if slurm_array_task_id else 0
    
    # Grab the current experiment from params.py
    experiment = experiments[task_id]
    experiment["TASK_ID"] = task_id
    set_run_params(experiment)
    
    create_experiment_dirs()
    initialize_logger()
    get_logger().info(f'Experiment {task_id + 1} started')
    get_logger().info(f'Experiment type: {"SPLINTERED" if experiment["IS_ENCODED"] else "BASELINE"}')
    
    try:
        run()
    except Exception as e:
        get_logger().exception(f'Experiment {task_id + 1} failed with error: \n{e}')
        exit(1)
        
    get_logger().info(f'Experiment {task_id + 1} finished')