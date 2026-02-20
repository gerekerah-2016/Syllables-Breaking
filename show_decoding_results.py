"""
Show FULL decoding results with vowels restored.
"""

import json
import re
from pathlib import Path
from src.language_utils.EthiopicUtils import EthiopicUtils


def main():
    # Initialize
    utils = EthiopicUtils()
    
    # Load decode map
    decode_map_path = Path("outputs/2026-02-17-geez-all_letters/results/splinter/new_unicode_chars_inverted.json")
    with open(decode_map_path, 'r', encoding='utf-8') as f:
        decode_map = json.load(f)
    
    print("\n" + "=" * 80)
    print("FULL DECODING WITH VOWELS RESTORED")
    print("=" * 80)
    
    # Test examples manually
    test_cases = [
        ("ወአተ", ["⟨1⟩", "⟨2⟩", "⟨3⟩"]),
        ("አሰረአለ", ["⟨1⟩", "⟨2⟩", "⟨8⟩", "⟨9⟩", "⟨10⟩"]),
        ("ዘነገረመ", ["⟨26⟩", "⟨30⟩"]),
    ]
    
    print("\n📝 Manual Decoding Tests:")
    for skeleton, tags in test_cases:
        decoded = utils.decode_from_splinter(skeleton, tags, decode_map)
        tag_str = " ".join(tags)
        print(f"\n  Skeleton: {skeleton}")
        print(f"  Tags: {tag_str}")
        print(f"  Decoded: {decoded}")
    
    # Read first few lines from tokenized file
    tokenized_path = Path("outputs/2026-02-17-geez-all_letters/tokenized_corpora/splintered_bpe_10000_tokenized.txt")
    
    print("\n" + "=" * 80)
    print("FIRST 10 LINES - FULL DECODING")
    print("=" * 80)
    
    with open(tokenized_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            
            line = line.strip()
            if not line:
                continue
            
            # Split into words
            parts = line.split()
            reconstructed = []
            
            j = 0
            while j < len(parts):
                word = parts[j]
                
                # Check if this word has tags
                if word.startswith('⟨') and word.endswith('⟩'):
                    # This is a tag - should be attached to previous word
                    if reconstructed:
                        reconstructed[-1] += word
                else:
                    # This is a word - look ahead for tags
                    full_word = word
                    while j + 1 < len(parts) and parts[j + 1].startswith('⟨'):
                        full_word += parts[j + 1]
                        j += 1
                    reconstructed.append(full_word)
                j += 1
            
            print(f"\nLine {i+1}:")
            for word in reconstructed:
                # Extract skeleton and tags
                skeleton = ''.join([c for c in word if utils.is_letter_in_language(c)])
                tags = re.findall(r'⟨\d+⟩', word)
                
                if tags:
                    decoded = utils.decode_from_splinter(skeleton, tags, decode_map)
                    print(f"  {word} → {decoded}")
                else:
                    print(f"  {word} → {word} (no tags)")

if __name__ == "__main__":
    main()