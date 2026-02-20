import os
from src.TextProcessorForDemo import TextProcessorForDemo
from src.language_utils.LanguageUtilsFactory import LanguageUtilsFactory
from src.logger import get_logger, initialize_logger
from src.params import set_run_params, get_dummy_experiment
from src.utils.path_utils import get_logs_dir
from src.utils.utils import (
    get_reductions_map_from_file,
    get_new_unicode_chars_map_from_file,
    get_new_unicode_chars_inverted_map_from_file
)

def demo_ethiopic(text: str, language: str = 'ge'):
    """Demo the splinter process for Ethiopic text"""
    
    # Setup
    experiment = get_dummy_experiment(f'2026-02-17-{language}-demo')
    set_run_params(experiment)
    os.makedirs(get_logs_dir(), exist_ok=True)
    initialize_logger()
    
    # Get language utils
    language_utils = LanguageUtilsFactory.get_by_language(language)
    get_logger().info(f'Language: {language_utils.__class__.__name__}')
    
    # Load maps (should exist from training)
    try:
        reductions_map = get_reductions_map_from_file()
        new_unicode_chars_map = get_new_unicode_chars_map_from_file()
        new_unicode_chars_inverted_map = get_new_unicode_chars_inverted_map_from_file()
    except FileNotFoundError:
        get_logger().error("Maps not found. Run training first with IS_ENCODED=True")
        return
    
    # Create processor
    processor = TextProcessorForDemo(
        language_utils,
        reductions_map,
        new_unicode_chars_map,
        new_unicode_chars_inverted_map
    )
    
    # Process
    get_logger().info(f'\nOriginal text: {text}')
    
    if hasattr(processor, 'process_with_stages'):
        # For Ethiopic, show decomposition
        decomposed, reduced, encoded = processor.process_with_stages(text)
        get_logger().info(f'\nDecomposed: {decomposed}')
        get_logger().info(f'\nReduced: {reduced}')
        get_logger().info(f'\nEncoded: {encoded}')
    else:
        non_encoded, encoded = processor.process(text)
        get_logger().info(f'\nNon-encoded: {non_encoded}')
        get_logger().info(f'\nEncoded: {encoded}')
    
    # Decode
    decoded, original = processor.undo_process(encoded)
    get_logger().info(f'\nDecoded: {decoded}')
    get_logger().info(f'\nBack to original: {original}')

if __name__ == '__main__':
    # Test with Ge'ez example from document
    demo_ethiopic("ባሱማ", 'ge')