"""
Save processed corpus to text file.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

from src.logger import get_logger
from src.utils.path_utils import get_corpus_path
from src.params import get_run_params


def save_corpus_as_text_file(text_processor, corpus_path: str, corpus_name: str):
    """
    Save processed corpus to text file.
    
    Args:
        text_processor: Text processor (baseline or encoded)
        corpus_path: Path to original corpus
        corpus_name: Name for output file
    """
    logger = get_logger()
    
    # Determine output type
    if get_run_params("IS_ENCODED"):
        output_name = "splintered"
    else:
        output_name = "baseline"
    
    output_path = get_corpus_path(output_name)
    
    logger.info(f"Saving {output_name} corpus to: {output_path}")
    
    # Load and process corpus
    from src.corpus.CorpusLoader import CorpusLoader
    loader = CorpusLoader(logger)
    words = loader.load(corpus_path, is_huggingface=get_run_params("IS_HUGGINGFACE"))
    
    # Process and save
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, word in enumerate(words):
            if not word or len(word.strip()) == 0:
                continue
            
            processed = text_processor.process(word)
            f.write(processed + '\n')
            
            if (i + 1) % 10000 == 0:
                logger.info(f"  Processed {i+1}/{len(words)} words")
    
    # Print summary statistics (only once)
    if hasattr(text_processor, 'print_summary'):
        text_processor.print_summary()
    
    # Also save original corpus
    original_path = get_corpus_path("original")
    with open(original_path, 'w', encoding='utf-8') as f:
        for word in words:
            if word and len(word.strip()) > 0:
                f.write(word + '\n')
    
    logger.info(f"✓ Saved {output_name} corpus ({len(words)} words)")