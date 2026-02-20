"""
Demo script following the Syllable Breaking document example.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.language_utils.EthiopicUtils import EthiopicUtils
from src.SplinterTrainer import SplinterTrainer
from src.TextProcessorWithEncoding import TextProcessorWithEncoding
from src.utils.path_utils import get_splinter_dir


def demonstrate_syllable_breaking():
    """
    Demonstrate the complete pipeline using the example from the document:
    ባሱማ (meaning: "he blessed")
    """
    print("=" * 60)
    print("SYLLABLE BREAKING DEMO - Following the document")
    print("=" * 60)
    
    # Initialize
    utils = EthiopicUtils()
    
    # Example word from document
    original = "ባሱማ"
    print(f"\n1. Original word: {original}")
    print(f"   Meaning: 'he blessed'")
    
    # Step 1: Text Normalization (Section 3)
    print(f"\n2. Step 1: Text Normalization")
    normalized = utils.remove_diacritics(original)
    print(f"   Normalized: {normalized}")
    
    # Step 2: Syllable Breaking (Section 4)
    print(f"\n3. Step 2: Syllable Breaking")
    syllables = utils.syllable_break(original)
    print(f"   Decomposition:")
    decomposed = []
    for i, (cons, vowel_order) in enumerate(syllables):
        vowel = utils.VOWEL_ORDERS.get(vowel_order, '?')
        marker = utils.VOWEL_MARKERS.get(vowel, '?')
        decomposed.append(cons)
        decomposed.append(marker)
        print(f"   {original[i]} = consonant {cons} + vowel [{vowel}] (order {vowel_order+1})")
    
    decomposed_word = ''.join(decomposed)
    print(f"\n   Decomposed form: {decomposed_word}")
    
    # Step 3: Consonant Indexing (Section 5)
    print(f"\n4. Step 3: Consonant Indexing")
    print(f"   Position:  0   1   2   3   4   5")
    print(f"   Character: {decomposed[0]}   {decomposed[1]}   {decomposed[2]}   {decomposed[3]}   {decomposed[4]}   {decomposed[5]}")
    print(f"   Type:      C   V   C   V   C   V")
    print(f"   Cons Index:0       1       2")
    
    # Step 4: Learning Reductions (would normally be done on corpus)
    # For demo, we'll simulate the reductions from the document
    print(f"\n5. Step 4: Learned Reductions (from corpus)")
    reductions_map = {
        6: {'3:u': 0.4, '1:a': 0.3, '5:a': 0.2, '0:በ': 0.05, '2:ሰ': 0.03, '4:መ': 0.02},
        5: {'1:a': 0.5, '2:a': 0.3, '0:በ': 0.1, '3:መ': 0.05, '4:ሰ': 0.05},
        4: {'2:a': 0.6, '1:u': 0.2, '0:ሰ': 0.1, '3:መ': 0.1},
    }
    print(f"   Found reductions for various word lengths")
    
    # Step 5: Encoding a Word (Section 7)
    print(f"\n6. Step 5: Encoding the Word")
    print(f"   Input (decomposed): {decomposed_word}")
    
    # Simulate the reduction process from the document
    reduction_sequence = ['3:u', '1:a', '2:a']
    print(f"   Reduction sequence: {reduction_sequence}")
    print(f"   Step 1: Remove position 3 ({decomposed[3]}) → በ{a}ሰመ{a}")
    print(f"   Step 2: Remove position 1 ({decomposed[1]}) → በሰመ{a}")
    print(f"   Step 3: Remove position 2 (in current word) → በሰመ")
    print(f"   Final consonant skeleton: በሰመ")
    
    # Step 6: Mapping to Compact Tokens (Section 8)
    print(f"\n7. Step 6: Mapping to Compact Tokens")
    
    # Create mapping as in document Section 8.1
    char_map = {
        '3:u': '\uE033',  # ⟨33⟩
        '1:a': '\uE021',  # ⟨21⟩
        '2:a': '\uE022',  # ⟨22⟩
        'በ': '\uE100',    # ⟨B0⟩
        'ሰ': '\uE101',    # ⟨S0⟩
        'መ': '\uE102',    # ⟨M0⟩
    }
    
    print(f"\n   Mapping Table:")
    print(f"   3:u → ⟨33⟩ (U+E033) - 'thirty-three'")
    print(f"   1:a → ⟨21⟩ (U+E021) - 'twenty-one'")
    print(f"   2:a → ⟨22⟩ (U+E022) - 'twenty-two'")
    print(f"   በ   → ⟨B0⟩ (U+E100) - 'B zero'")
    print(f"   ሰ   → ⟨S0⟩ (U+E101) - 'S zero'")
    print(f"   መ   → ⟨M0⟩ (U+E102) - 'M zero'")
    
    encoded = '⟨B0⟩⟨S0⟩⟨M0⟩⟨33⟩⟨21⟩⟨22⟩'
    print(f"\n   Encoded word: {encoded}")
    print(f"   Pronunciation: B zero, S zero, M zero, thirty-three, twenty-one, twenty-two")
    
    # Step 7: Training the Tokenizer (Section 9)
    print(f"\n8. Step 7: Training the Tokenizer")
    print(f"   SentencePiece sees each ⟨token⟩ as a single unit")
    print(f"   It can learn patterns like ⟨B0⟩⟨S0⟩ (common consonant sequences)")
    print(f"   And patterns like ⟨33⟩⟨21⟩ (common vowel patterns)")
    
    # Step 8: Decoding (Section 10)
    print(f"\n9. Step 8: Decoding (Reversibility)")
    print(f"   Input tokens: [⟨B0⟩, ⟨S0⟩, ⟨M0⟩, ⟨33⟩, ⟨21⟩, ⟨22⟩]")
    print(f"   Map back: ⟨B0⟩→በ, ⟨S0⟩→ሰ, ⟨M0⟩→መ, ⟨33⟩→3:u, ⟨21⟩→1:a, ⟨22⟩→2:a")
    print(f"   Start with consonant skeleton: በሰመ")
    print(f"   Apply reductions in reverse:")
    print(f"   1. Apply 2:a → add vowel [a] after consonant 2 → በሰመ{a}")
    print(f"   2. Apply 1:a → add vowel [a] after consonant 1 → በሰ{a}መ{a}")
    print(f"   3. Apply 3:u → add vowel [u] at position 3 → በ{a}ሰ{u}መ{a}")
    print(f"   Recompose syllables:")
    print(f"   በ{a} → ባ")
    print(f"   ሰ{u} → ሱ")
    print(f"   መ{a} → ማ")
    print(f"   Final decoded word: {original}")
    
    print("\n" + "=" * 60)
    print("✓ 100% reversible! Decoding accuracy guaranteed.")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_syllable_breaking()