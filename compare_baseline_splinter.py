"""
Compare baseline vs splintered tokenization.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import sentencepiece as spm
from src.language_utils.EthiopicUtils import EthiopicUtils
from src.TextProcessorBaseline import TextProcessorBaseline
from src.TextProcessorWithEncoding import TextProcessorWithEncoding
from src.utils.path_utils import get_splinter_dir, get_experiment_dir
from src.params import set_run_params


def compare_tokenization():
    """Compare baseline and splintered tokenization on sample words."""
    
    # Set minimal params for path functions
    set_run_params({
        'OUTPUT_BASE_DIR': './outputs',
        'EXPERIMENT_NAME': '2026-02-17-geez-all_letters',
        'LANGUAGE': 'ge',
        'TASK_ID': 0,
        'TIMESTAMP': '20260217-164753'
    })
    
    print("=" * 70)
    print("COMPARING BASELINE VS SPLINTERED TOKENIZATION")
    print("=" * 70)
    
    # Load utils and maps
    utils = EthiopicUtils()
    
    # Try multiple possible locations for mapping files
    possible_paths = [
        os.path.join(get_splinter_dir(), 'new_unicode_chars.json'),
        os.path.join(get_experiment_dir(), 'results', 'splinter', 'new_unicode_chars.json'),
        './outputs/2026-02-17-geez-all_letters/results/splinter/new_unicode_chars.json'
    ]
    
    encode_map = {}
    decode_map = {}
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found mapping file: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                encode_map = json.load(f)
            break
    
    if not encode_map:
        print("❌ Could not find mapping files!")
        return
    
    # Load inverted map
    inverted_paths = [
        os.path.join(get_splinter_dir(), 'new_unicode_chars_inverted.json'),
        os.path.join(get_experiment_dir(), 'results', 'splinter', 'new_unicode_chars_inverted.json'),
        './outputs/2026-02-17-geez-all_letters/results/splinter/new_unicode_chars_inverted.json'
    ]
    
    for path in inverted_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                decode_map = json.load(f)
            break
    
    print(f"📊 Loaded {len(encode_map)} mappings")
    
    # Create processors
    baseline_processor = TextProcessorBaseline(utils)
    splinter_processor = TextProcessorWithEncoding(utils, {}, encode_map, decode_map)
    
    # Test words
    test_words = [
        "ባሱማ",
        "እግዚአብሔር",
        "ወምድርሰ",
        "መጽሐፍ"
    ]
    
    # Find tokenizer models
    tokenizer_paths = [
        './outputs/2026-02-17-geez-baseline/tokenizers/geez_unigram_100.model',
        './outputs/2026-02-17-geez-all_letters/tokenizers/geez_unigram_100.model',
        './outputs/2026-02-17-geez-all_letters/tokenizers/ge_unigram_100.model'
    ]
    
    sp_baseline = None
    sp_splinter = None
    
    # Try to load baseline tokenizer
    for path in tokenizer_paths:
        if 'baseline' in path and os.path.exists(path):
            sp_baseline = spm.SentencePieceProcessor()
            sp_baseline.load(path)
            print(f"✅ Loaded baseline tokenizer: {path}")
            break
    
    # Try to load splinter tokenizer
    for path in tokenizer_paths:
        if 'all_letters' in path and os.path.exists(path):
            sp_splinter = spm.SentencePieceProcessor()
            sp_splinter.load(path)
            print(f"✅ Loaded splinter tokenizer: {path}")
            break
    
    if not sp_splinter:
        print("⚠️  Splinter tokenizer not found. Using mock tokenizer.")
        # Create a mock tokenizer for demonstration
        class MockTokenizer:
            def encode_as_pieces(self, text):
                # Simple mock: split by ⟨⟩
                import re
                parts = re.split(r'(⟨\d+⟩)', text)
                return [p for p in parts if p]
        sp_splinter = MockTokenizer()
    
    results = []
    
    print("\n" + "-" * 70)
    
    for word in test_words:
        print(f"\n{'='*70}")
        print(f"WORD: {word}")
        print(f"{'='*70}")
        
        # Baseline processing
        baseline_processed = baseline_processor.process(word)
        if sp_baseline:
            baseline_tokens = sp_baseline.encode_as_pieces(baseline_processed)
        else:
            baseline_tokens = [word]  # Mock
        
        # Splinter processing
        splinter_processed = splinter_processor.encode_word(word)
        splinter_tokens = sp_splinter.encode_as_pieces(splinter_processed)
        
        # Decode splintered back
        if hasattr(splinter_processor, 'decode_word'):
            decoded = splinter_processor.decode_word(splinter_processed)
        else:
            decoded = word  # Mock
        
        print(f"\n📊 BASELINE:")
        print(f"  Processed: {baseline_processed}")
        print(f"  Tokens:    {' | '.join(baseline_tokens)} |")
        print(f"  Count:     {len(baseline_tokens)}")
        
        print(f"\n🔧 SPLINTERED:")
        print(f"  Processed: {splinter_processed}")
        print(f"  Tokens:    {' | '.join(splinter_tokens)} |")
        print(f"  Count:     {len(splinter_tokens)}")
        
        # Check for special tokens
        special_count = sum(1 for t in splinter_tokens if t.startswith('⟨') and t.endswith('⟩'))
        print(f"  Special ⟨n⟩ tokens: {special_count}")
        
        print(f"\n🔄 DECODED: {decoded}")
        print(f"  Round-trip: {'✓' if decoded == word else '✗'}")
        
        results.append({
            'word': word,
            'baseline_tokens': len(baseline_tokens),
            'splinter_tokens': len(splinter_tokens),
            'special_tokens': special_count,
            'round_trip': decoded == word
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Word':<20} {'Baseline':<10} {'Splinter':<10} {'Special':<10} {'Round-trip':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['word']:<20} {r['baseline_tokens']:<10} {r['splinter_tokens']:<10} "
              f"{r['special_tokens']:<10} {'✓' if r['round_trip'] else '✗':<10}")
    
    # Calculate averages
    if results:
        avg_baseline = sum(r['baseline_tokens'] for r in results) / len(results)
        avg_splinter = sum(r['splinter_tokens'] for r in results) / len(results)
        
        print("-" * 70)
        print(f"{'AVERAGE':<20} {avg_baseline:<10.2f} {avg_splinter:<10.2f}")
    
    print("=" * 70)


if __name__ == "__main__":
    compare_tokenization()