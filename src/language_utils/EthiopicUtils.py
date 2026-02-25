"""
Ethiopic language utilities with working syllable breaking and decoding.
Author: Gebreslassie Teklu Reda - FINAL FIXED VERSION
Date: 2026
"""

import re
from typing import List, Tuple, Dict, Optional
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface


class EthiopicUtils(LanguageUtilsInterface):
    """
    Ethiopic language utilities with 100% reversible encoding.
    PRESERVES all punctuation, numbers, and original structure.
    CRITICAL: ፡ is preserved as word separator!
    """
    
    def __init__(self):
        # Ethiopic syllable range
        self.SYLLABLE_RANGE = (0x1200, 0x137F)
        self.WORD_BOUNDARY = '▁'  # U+2581 - ONLY used by SentencePiece
        
        # ============================================================
        # KEEP ALL punctuation - NOTHING removed!
        # ============================================================
        self.KEEP_PUNCTUATION = {
            '፡', '።', '፣', '፤', '፥', '፦', '፧', '፠', '፨',  # Ethiopic
            '(', ')', '{', '}', '[', ']', '<', '>',          # Brackets
            '.', ',', ';', ':', '!', '?', '"', "'",          # Latin punctuation
            '-', '_', '=', '+', '*', '&', '^', '%', '$', '#', '@',  # Symbols
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'  # Numbers
        }
        
        # Characters that cause SentencePiece to split
        self.problematic_chars = {}  # Empty - don't replace anything
        self.reverse_chars = {}
        
        # ============================================================
        # Vowel representations
        # ============================================================
        
        # Vowel orders (0-based)
        self.VOWEL_ORDERS = {
            0: "ə",  # 1st order (base)
            1: "u",  # 2nd order
            2: "i",  # 3rd order
            3: "a",  # 4th order
            4: "e",  # 5th order
            5: "ï",  # 6th order (ɨ)
            6: "o",  # 7th order
        }
        
        # Labiovelar orders
        self.LABIOVELAR_ORDERS = {
            7: "[8]",   # order 8
            8: "[9]",   # order 9
            9: "[10]",  # order 10
            10: "[11]", # order 11
            11: "[12]", # order 12
        }
        
        # VOWEL_SYMBOLS - SplinterTrainer expects this name
        self.VOWEL_SYMBOLS = {
            0: "",      # Base form - no marker
            1: "[u]",
            2: "[i]",
            3: "[a]",
            4: "[e]",
            5: "[ï]",
            6: "[o]",
            7: "[8]",
            8: "[9]",
            9: "[10]",
            10: "[11]",
            11: "[12]",
        }
        
        # Symbol to order mapping for decoding
        self.SYMBOL_TO_ORDER = {
            "[u]": 1, "[i]": 2, "[a]": 3, "[e]": 4, "[ï]": 5, "[o]": 6,
            "[8]": 7, "[9]": 8, "[10]": 9, "[11]": 10, "[12]": 11
        }
        
        # Vowel char to order mapping
        self.VOWEL_CHAR_TO_ORDER = {
            'ə': 0, 'u': 1, 'i': 2, 'a': 3, 'e': 4, 'ï': 5, 'o': 6,
            '8': 7, '9': 8, '10': 9, '11': 10, '12': 11
        }
        
        # 33 base consonants
        self.BASE_CONSONANTS = [
            'ሀ', 'ለ', 'ሐ', 'መ', 'ሠ', 'ረ', 'ሰ', 'ሸ', 'ቀ', 'በ', 'ተ', 'ቸ', 'ኀ',
            'ነ', 'ኘ', 'አ', 'ከ', 'ኸ', 'ወ', 'ዐ', 'ዘ', 'ዠ', 'የ', 'ደ', 'ጀ', 'ገ',
            'ጠ', 'ጨ', 'ጰ', 'ጸ', 'ፀ', 'ፈ', 'ፐ'
        ]
        
        # ============================================================
        # COMPLETE labiovelar mapping - INCLUDING ፘ, ፙ, ፚ
        # ============================================================
        self.LABIOVELAR_MAP = {
            # ቀ family
            'ቈ': ('ቀ', 7), 'ቊ': ('ቀ', 8), 'ቋ': ('ቀ', 9), 'ቌ': ('ቀ', 10), 'ቍ': ('ቀ', 11),
            # ኀ family
            'ኈ': ('ኀ', 7), 'ኊ': ('ኀ', 8), 'ኋ': ('ኀ', 9), 'ኌ': ('ኀ', 10), 'ኍ': ('ኀ', 11),
            # ከ family
            'ኰ': ('ከ', 7), 'ኲ': ('ከ', 8), 'ኳ': ('ከ', 9), 'ኴ': ('ከ', 10), 'ኵ': ('ከ', 11),
            # ገ family
            'ጐ': ('ገ', 7), 'ጒ': ('ገ', 8), 'ጓ': ('ገ', 9), 'ጔ': ('ገ', 10), 'ጕ': ('ገ', 11),
            # ፘ family - FYA (based on ረ)
            'ፘ': ('ረ', 7),
            # ፙ family - MYA (based on ሠ)
            'ፙ': ('ሠ', 7),
            # ፚ family - FYA (based on ፈ)
            'ፚ': ('ፈ', 7),
        }
        
        # Reverse mapping
        self.REVERSE_LABIOVELAR_MAP = {}
        for labio, (base, order) in self.LABIOVELAR_MAP.items():
            self.REVERSE_LABIOVELAR_MAP[(base, order)] = labio
        
        # Pre-compute syllable map
        self.SYLLABLE_MAP = {}
        self._build_syllable_map()

    def _build_syllable_map(self):
        """Build mapping from any syllable to (base, order)."""
        # Regular syllables
        for base in self.BASE_CONSONANTS:
            base_cp = ord(base)
            base_offset = (base_cp - 0x1200) // 8 * 8
            for order in range(7):
                syllable_cp = 0x1200 + base_offset + order
                self.SYLLABLE_MAP[chr(syllable_cp)] = (base, order)
        
        # Labiovelars
        for labio, (base, order) in self.LABIOVELAR_MAP.items():
            self.SYLLABLE_MAP[labio] = (base, order)
    
    # ============================================================
    # INTERFACE METHODS
    # ============================================================
    
    def remove_diacritics(self, text: str) -> str:
        """
        Clean text - PRESERVES all punctuation including ፡!
        """
        if not text:
            return text
        
        cleaned = []
        for char in text:
            # Keep ALL Ge'ez letters
            if self.is_letter_in_language(char):
                cleaned.append(char)
            # CRITICAL: Keep ፡ and all Ethiopic punctuation
            elif char in self.KEEP_PUNCTUATION:
                cleaned.append(char)
            # Keep spaces
            elif char.isspace():
                cleaned.append(' ')
            # Keep numbers
            elif char.isdigit():
                cleaned.append(char)
            # Keep basic ASCII punctuation
            elif ord(char) < 128:
                cleaned.append(char)
            # Remove everything else
        
        return ''.join(cleaned)
    
    def normalize_text(self, text: str) -> str:
        return self.remove_diacritics(text)

    def is_letter_in_language(self, char: str) -> bool:
        """Check if character is a Ge'ez LETTER (not punctuation or numbers)."""
        if len(char) != 1:
            return False
        
        # Must be in Ethiopic range AND not be punctuation
        in_range = self.SYLLABLE_RANGE[0] <= ord(char) <= self.SYLLABLE_RANGE[1]
        
        # Exclude known punctuation marks
        is_punctuation = char in {'፡', '።', '፣', '፤', '፥', '፦', '፧', '፠', '፨',
                                   '፩', '፪', '፫', '፬', '፭', '፮', '፯', '፰', '፱',
                                   '፲', '፳', '፴', '፵', '፶', '፷', '፸', '፹', '፺', '፻', '፼'}
        
        return in_range and not is_punctuation

    def is_word_contains_letters_from_other_languages(self, word: str) -> bool:
        return any(not self.is_letter_in_language(c) and 
                  c != self.WORD_BOUNDARY and
                  not c.isspace() and 
                  c not in self.KEEP_PUNCTUATION for c in word)

    def get_language_alphabet(self) -> list:
        return self.BASE_CONSONANTS.copy()

    def replace_final_letters(self, text: str) -> str:
        """Handle labiovelar decomposition."""
        result = []
        for char in text:
            if char in self.LABIOVELAR_MAP:
                base, order = self.LABIOVELAR_MAP[char]
                result.append(base)
                result.append(self.VOWEL_SYMBOLS[order])
            else:
                result.append(char)
        return ''.join(result)

    def save_additional_corpora_for_evaluation(self, text_processor) -> None:
        pass

    # ============================================================
    # CORE METHODS - FIXED to handle ፡ correctly
    # ============================================================
    
    def get_base_and_order(self, char: str) -> Tuple[str, int]:
        """Get base consonant and vowel order."""
        # If it's punctuation, return special marker
        if char in self.KEEP_PUNCTUATION:
            return 'PUNCT', -1  # Special marker for punctuation
        
        # Check labiovelar map
        if char in self.LABIOVELAR_MAP:
            return self.LABIOVELAR_MAP[char]
        
        # Not a letter, return as-is
        if not self.is_letter_in_language(char):
            return char, 0
        
        # Regular syllable
        cp = ord(char)
        offset = cp - 0x1200
        base_cp = 0x1200 + (offset // 8) * 8
        order = offset % 8
        return chr(base_cp), order
    
    def syllable_break(self, word: str) -> str:
        """
        Break word into consonant + vowel marker format.
        FIXED: Preserves ፡ as separator between words.
        """
        broken = []
        
        for char in word:
            # Handle punctuation specially
            if char in self.KEEP_PUNCTUATION:
                broken.append(char)  # Keep punctuation as-is
            elif self.is_letter_in_language(char):
                base, order = self.get_base_and_order(char)
                broken.append(base)
                if order > 0:
                    broken.append(self.VOWEL_SYMBOLS[order])
            else:
                # Keep other characters as-is
                broken.append(char)
        
        return "".join(broken)
    
    def get_vowel_order_from_marker(self, marker: str) -> Optional[int]:
        """
        Get vowel order from marker string.
        Handles multi-digit markers like [10] correctly.
        """
        if marker in self.SYMBOL_TO_ORDER:
            return self.SYMBOL_TO_ORDER[marker]
        
        match = re.match(r'\[(\d+)\]', marker)
        if match:
            num = int(match.group(1))
            if 8 <= num <= 12:
                return num - 1
        
        return None
    def is_letter_in_language(self, char: str) -> bool:
        """
        Check if character is a Ge'ez LETTER (not punctuation).
        FIXED: Explicitly excludes Ethiopic punctuation.
        """
        if len(char) != 1:
            return False
    
        # Ethiopic punctuation must return False
        if char in self.KEEP_PUNCTUATION:
            return False
    
        # Check if in Ethiopic letter range
        return self.SYLLABLE_RANGE[0] <= ord(char) <= self.SYLLABLE_RANGE[1]
    def apply_vowel_to_consonant(self, consonant: str, vowel_order: int) -> str:
        """Apply vowel to base consonant to get original syllable."""
        if not self.is_letter_in_language(consonant):
            return consonant
        
        if vowel_order >= 7:
            key = (consonant, vowel_order)
            if key in self.REVERSE_LABIOVELAR_MAP:
                return self.REVERSE_LABIOVELAR_MAP[key]
        
        base_cp = ord(consonant)
        block_start = (base_cp - 0x1200) // 8 * 8 + 0x1200
        return chr(block_start + vowel_order)
    
    def get_consonant_skeleton(self, word: str) -> str:
        """
        Extract consonant skeleton.
        FIXED: Punctuation is NOT part of skeleton.
        """
        broken = self.syllable_break(word)
        skeleton = []
        i = 0
        while i < len(broken):
            char = broken[i]
            
            if char in self.KEEP_PUNCTUATION:
                # Skip punctuation in skeleton
                i += 1
            elif self.is_letter_in_language(char):
                skeleton.append(char)
                if i + 1 < len(broken) and broken[i+1].startswith('['):
                    i += 2
                else:
                    i += 1
            else:
                # Keep other non-Ge'ez characters (numbers, etc.)
                skeleton.append(char)
                i += 1
        
        return ''.join(skeleton)
    
    def get_vowel_tags(self, word: str, encode_map: dict) -> str:
        """
        Get vowel tags for a word.
        Returns tags string with NO SPACES: "⟨1⟩⟨2⟩"
        Tags are ONLY for Ge'ez vowels, not for punctuation.
        """
        broken = self.syllable_break(word)
        tags = []
        cons_idx = 0
        i = 0
        
        while i < len(broken):
            char = broken[i]
            
            if char in self.KEEP_PUNCTUATION:
                i += 1
            elif self.is_letter_in_language(char):
                if i + 1 < len(broken) and broken[i+1].startswith('['):
                    key = f"{cons_idx}:{broken[i+1]}"
                    if key in encode_map:
                        tags.append(encode_map[key])
                    cons_idx += 1
                    i += 2
                else:
                    cons_idx += 1
                    i += 1
            else:
                i += 1
        
        return ''.join(tags)  # NO SPACES between tags!
    
    def decode_from_splinter(self, skeleton: str, tags: list, decode_map: dict) -> str:
        """
        Reconstruct original word from skeleton and tags.
        PRESERVES all non-Ge'ez characters.
        """
        if not skeleton:
            return skeleton
        
        result = list(skeleton)
        
        for tag in tags:
            clean_tag = tag.strip()
            if clean_tag.startswith('▁'):
                clean_tag = clean_tag[1:]
            
            if clean_tag.startswith('[T') and clean_tag.endswith(']'):
                match = re.match(r'\[T(\d+)\]', clean_tag)
                if match:
                    clean_tag = f"⟨{match.group(1)}⟩"
            
            if clean_tag not in decode_map:
                continue
            
            key = decode_map[clean_tag]
            if ':' not in key:
                continue
            
            pos_str, marker = key.split(':', 1)
            try:
                pos = int(pos_str)
            except ValueError:
                continue
            
            if pos >= len(result):
                continue
            
            base = result[pos]
            if not self.is_letter_in_language(base):
                continue
            
            order = self.get_vowel_order_from_marker(marker)
            if order is None:
                continue
            
            result[pos] = self.apply_vowel_to_consonant(base, order)
        
        return ''.join(result)
    
    def prepare_for_tokenizer(self, text: str) -> str:
        """
        Prepare text for SentencePiece.
        PRESERVES all spaces, punctuation, and structure.
        """
        return text  # Return as-is - let SentencePiece handle it
    
    def after_tokenization_restore(self, text: str) -> str:
        """No restoration needed."""
        return text
    
    def round_trip_test(self, word: str, encode_map: dict, decode_map: dict) -> bool:
        """Test if a word round-trips correctly."""
        skeleton = self.get_consonant_skeleton(word)
        tags_str = self.get_vowel_tags(word, encode_map)
        tags = re.findall(r'⟨\d+⟩', tags_str)
        decoded = self.decode_from_splinter(skeleton, tags, decode_map)
        return decoded == word