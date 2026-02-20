"""
Ethiopic language utilities with working syllable breaking and decoding.
Includes labiovelar support.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import re
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface


class EthiopicUtils(LanguageUtilsInterface):
    """
    Ethiopic language utilities with correct syllable breaking and decoding.
    Handles the 33 consonants × 7 vowels + labiovelars (orders 8-12).
    """
    
    def __init__(self):
        # Ethiopic syllable range (U+1200 to U+137F)
        self.SYLLABLE_RANGE = (0x1200, 0x137F)
        
        # Word boundary marker (used in tokenization)
        self.WORD_BOUNDARY = '▁'  # U+2581
        
        # Standard Ge'ez punctuation to keep
        self.KEEP_PUNCTUATION = {'፡', '።', '፣', '፤', '፥', '፦'}
        
        # Punctuation to remove (including ፠)
        self.REMOVE_PUNCTUATION = {'፠', '፧', '፨', '፩', '፪', '፫', '፬', '፭', '፮', '፯', '፰', '፱', 
                                   '፲', '፳', '፴', '፵', '፶', '፷', '፸', '፹', '፺', '፻', '፼'}
        
        # ============================================================
        # Characters that cause SentencePiece to split
        # ============================================================
        self.problematic_chars = {
            '(': '｟', ')': '｠',
            '[': '｢', ']': '｣',
            '{': '｛', '}': '｝',
        }
        
        # Reverse mapping for restoration
        self.reverse_chars = {v: k for k, v in self.problematic_chars.items()}
        
        # ============================================================
        # Vowel representations (including labiovelars)
        # ============================================================
        
        # Standard vowel orders (0-6)
        self.VOWEL_ORDERS = {
            0: "ə",  # 1st order (base)
            1: "u",  # 2nd order
            2: "i",  # 3rd order
            3: "a",  # 4th order
            4: "e",  # 5th order
            5: "ï",  # 6th order
            6: "o",  # 7th order
        }
        
        # Labiovelar vowel orders (7-11) - representing orders 8-12 in 1-based
        self.LABIOVELAR_ORDERS = {
            7: "[8]",   # order 8
            8: "[9]",   # order 9
            9: "[10]",  # order 10
            10: "[11]", # order 11
            11: "[12]", # order 12
        }
        
        # Combined vowel symbols for display (all orders 0-11)
        self.VOWEL_SYMBOLS = {
            0: "[ə]",
            1: "[u]",
            2: "[i]",
            3: "[a]",
            4: "[e]",
            5: "[ï]",
            6: "[o]",
            7: "[8]",   # labiovelar markers
            8: "[9]",
            9: "[10]",
            10: "[11]",
            11: "[12]",
        }
        
        # Reverse mapping for decoding (symbol -> order)
        self.SYMBOL_TO_ORDER = {
            "[ə]": 0, "[u]": 1, "[i]": 2, "[a]": 3,
            "[e]": 4, "[ï]": 5, "[o]": 6,
            "[8]": 7, "[9]": 8, "[10]": 9, "[11]": 10, "[12]": 11
        }
        
        # Direct vowel character to order mapping
        self.VOWEL_CHAR_TO_ORDER = {
            'ə': 0, 'u': 1, 'i': 2, 'a': 3, 'e': 4, 'ï': 5, 'o': 6,
            'ɨ': 5,  # Alternative representation of the 6th order vowel
            '8': 7, '9': 8, '10': 9, '11': 10, '12': 11  # For numeric markers
        }
        
        # 33 base consonants (1st order forms)
        self.BASE_CONSONANTS = [
            'ሀ', 'ለ', 'ሐ', 'መ', 'ሠ', 'ረ', 'ሰ', 'ሸ', 'ቀ', 'በ', 'ተ', 'ቸ', 'ኀ',
            'ነ', 'ኘ', 'አ', 'ከ', 'ኸ', 'ወ', 'ዐ', 'ዘ', 'ዠ', 'የ', 'ደ', 'ጀ', 'ገ',
            'ጠ', 'ጨ', 'ጰ', 'ጸ', 'ፀ', 'ፈ', 'ፐ'
        ]
        
        # ============================================================
        # Labiovelar mapping (FIXED - maps to base + order)
        # ============================================================
        
        # Labiovelar mapping - format: labiovelar_char -> (base_consonant, vowel_order)
        self.LABIOVELAR_MAP = {
            # ቀ family (ቀ + labiovelar)
            'ቈ': ('ቀ', 7),   # ቈ = ቀ + order 8 (marker [8])
            'ቊ': ('ቀ', 8),   # ቊ = ቀ + order 9 (marker [9])
            'ቋ': ('ቀ', 9),   # ቋ = ቀ + order 10 (marker [10])
            'ቌ': ('ቀ', 10),  # ቌ = ቀ + order 11 (marker [11])
            'ቍ': ('ቀ', 11),  # ቍ = ቀ + order 12 (marker [12])
            
            # ኀ family (ኀ + labiovelar)
            'ኈ': ('ኀ', 7),   # ኈ = ኀ + [8]
            'ኊ': ('ኀ', 8),   # ኊ = ኀ + [9]
            'ኋ': ('ኀ', 9),   # ኋ = ኀ + [10]
            'ኌ': ('ኀ', 10),  # ኌ = ኀ + [11]
            'ኍ': ('ኀ', 11),  # ኍ = ኀ + [12]
            
            # ከ family (ከ + labiovelar)
            'ኰ': ('ከ', 7),   # ኰ = ከ + [8]
            'ኲ': ('ከ', 8),   # ኲ = ከ + [9]
            'ኳ': ('ከ', 9),   # ኳ = ከ + [10]
            'ኴ': ('ከ', 10),  # ኴ = ከ + [11]
            'ኵ': ('ከ', 11),  # ኵ = ከ + [12]
            
            # ገ family (ገ + labiovelar)
            'ጐ': ('ገ', 7),   # ጐ = ገ + [8]
            'ጒ': ('ገ', 8),   # ጒ = ገ + [9]
            'ጓ': ('ገ', 9),   # ጓ = ገ + [10]
            'ጔ': ('ገ', 10),  # ጔ = ገ + [11]
            'ጕ': ('ገ', 11),  # ጕ = ገ + [12]
        }
        
        # Reverse mapping for reconstruction (base, order) -> labiovelar
        self.REVERSE_LABIOVELAR_MAP = {}
        for labio, (base, order) in self.LABIOVELAR_MAP.items():
            self.REVERSE_LABIOVELAR_MAP[(base, order)] = labio
        
        # ============================================================
        # Pre-computed syllable mapping for faster lookups
        # ============================================================
        self.SYLLABLE_MAP = {}
        self._build_syllable_map()

    def _build_syllable_map(self):
        """
        Build a mapping from any syllable to its (base_consonant, vowel_order).
        Includes both regular syllables and labiovelars.
        """
        # Regular syllables (orders 0-6)
        for base in self.BASE_CONSONANTS:
            base_cp = ord(base)
            base_offset = (base_cp - 0x1200) // 8 * 8
            
            for order in range(7):
                syllable_cp = 0x1200 + base_offset + order
                syllable = chr(syllable_cp)
                self.SYLLABLE_MAP[syllable] = (base, order)
        
        # Labiovelar syllables (orders 7-11)
        for labio, (base, order) in self.LABIOVELAR_MAP.items():
            self.SYLLABLE_MAP[labio] = (base, order)
    
    # ============================================================
    # REQUIRED INTERFACE METHODS
    # ============================================================
    
    def remove_diacritics(self, text: str) -> str:
        """
        Normalize text by removing unwanted punctuation like ፠.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text with only allowed characters
        """
        if not text:
            return text
            
        cleaned = []
        for char in text:
            # Keep Ethiopic letters
            if self.is_letter_in_language(char):
                cleaned.append(char)
            # Keep standard Ge'ez punctuation
            elif char in self.KEEP_PUNCTUATION:
                cleaned.append(char)
            # Keep word boundary marker
            elif char == self.WORD_BOUNDARY:
                cleaned.append(char)
            # Keep whitespace
            elif char.isspace():
                cleaned.append(char)
            # Remove everything else (including ፠, numbers, etc.)
            else:
                continue
        
        return ''.join(cleaned)
    
    def normalize_text(self, text: str) -> str:
        """
        Alias for remove_diacritics for consistency.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        return self.remove_diacritics(text)

    def is_letter_in_language(self, char: str) -> bool:
        """Check if character is in Ethiopic range."""
        if len(char) != 1:
            return False
        return self.SYLLABLE_RANGE[0] <= ord(char) <= self.SYLLABLE_RANGE[1]

    def is_word_contains_letters_from_other_languages(self, word: str) -> bool:
        """Check if word contains non-Ethiopic characters."""
        return any(not self.is_letter_in_language(c) and 
                  c != self.WORD_BOUNDARY and
                  not c.isspace() and 
                  c not in self.KEEP_PUNCTUATION for c in word)

    def get_language_alphabet(self) -> list:
        """Return all 33 base consonants."""
        return self.BASE_CONSONANTS.copy()

    def replace_final_letters(self, text: str) -> str:
        """
        Handle labiovelar decomposition - UPDATED to use proper mapping.
        Now decomposes labiovelars to base + vowel marker.
        """
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
        """Save additional corpora for evaluation."""
        pass

    # ============================================================
    # CORE ETHIOPIC METHODS
    # ============================================================
    
    def get_base_and_order(self, char: str):
        """
        Decompose a character into its base consonant and vowel order.
        
        Args:
            char: Ge'ez character
            
        Returns:
            Tuple of (base_consonant, vowel_order)
        """
        # Check if it's a labiovelar
        if char in self.LABIOVELAR_MAP:
            return self.LABIOVELAR_MAP[char]
        
        # Regular syllable
        if not self.is_letter_in_language(char):
            return char, 0
        
        cp = ord(char)
        offset = cp - 0x1200
        base_cp = 0x1200 + (offset // 8) * 8
        order = offset % 8
        return chr(base_cp), order
    
    def syllable_break(self, word: str) -> str:
        """
        Break Ge'ez into Consonant + [vowel] marker.
        
        This implements Section 4 of the document.
        Example: ባሱማ → በ[a]ሰ[u]መ[a]
        Example with labiovelars: ቋንቋ → ቀ[10]ንቀ[10]
        
        Args:
            word: Original Ge'ez word
            
        Returns:
            Decomposed form with vowel markers
        """
        broken = []
        
        for char in word:
            # Word boundary marker - preserve unchanged
            if char == self.WORD_BOUNDARY:
                broken.append(char)
                continue
            
            # Ge'ez syllable - decompose
            if self.is_letter_in_language(char):
                base, order = self.get_base_and_order(char)
                broken.append(base)
                if order > 0:  # Only add marker if not 1st order
                    broken.append(self.VOWEL_SYMBOLS[order])
            else:
                # Not Ge'ez - leave unchanged
                broken.append(char)
        
        return "".join(broken)

    def get_vowel_order_from_marker(self, marker: str) -> int:
        """
        Get vowel order from various marker formats.
        Now handles labiovelar markers like [8], [9], [10], [11], [12].
        """
        # Method 1: Direct lookup in SYMBOL_TO_ORDER
        if marker in self.SYMBOL_TO_ORDER:
            return self.SYMBOL_TO_ORDER[marker]
        
        # Method 2: Try stripping brackets and check
        clean = marker.strip('[]')
        
        # Check if it's a number (for labiovelars)
        if clean.isdigit():
            num = int(clean)
            if 8 <= num <= 12:
                return num - 1  # Convert 8→7, 9→8, 10→9, 11→10, 12→11
        
        # Check vowel char map
        if clean in self.VOWEL_CHAR_TO_ORDER:
            return self.VOWEL_CHAR_TO_ORDER[clean]
        
        # Method 3: Try direct match with VOWEL_ORDERS
        for order, vowel in self.VOWEL_ORDERS.items():
            if vowel == marker or vowel == clean:
                return order
        
        return None

    def apply_vowel_to_consonant(self, consonant: str, vowel_order: int) -> str:
        """
        Apply a vowel to a base consonant.
        Now handles labiovelar orders (7-11).
        
        Args:
            consonant: Base consonant (1st order)
            vowel_order: Vowel order (0-11)
            
        Returns:
            Syllable with the applied vowel
        """
        if not self.is_letter_in_language(consonant):
            return consonant
        
        # Check if this is a labiovelar order (7-11)
        if vowel_order >= 7:
            # Look up in reverse map
            key = (consonant, vowel_order)
            if key in self.REVERSE_LABIOVELAR_MAP:
                return self.REVERSE_LABIOVELAR_MAP[key]
        
        # Regular vowel (0-6)
        base_cp = ord(consonant)
        block_start = (base_cp - 0x1200) // 8 * 8 + 0x1200
        return chr(block_start + vowel_order)

    def decode_from_splinter(self, skeleton: str, tags: list, decode_map: dict) -> str:
        """
        Reconstruct original Ge'ez from skeleton and ⟨n⟩ tags.
        Now handles labiovelar orders.
        """
        # Convert skeleton to list of characters for manipulation
        result_chars = list(skeleton)
        
        # Process each tag in order
        for tag in tags:
            if tag not in decode_map:
                continue
                
            # Get the reduction key (e.g., "0:[a]" or "0:[10]")
            key = decode_map[tag]
            
            if ':' not in key:
                continue
                
            # Split into index and vowel marker
            idx_str, vowel_marker = key.split(":", 1)
            
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            
            if idx >= len(result_chars):
                continue
            
            # Get the base consonant at this position
            base_char = result_chars[idx]
            
            if not self.is_letter_in_language(base_char):
                continue
            
            # Determine vowel order from marker
            order = self.get_vowel_order_from_marker(vowel_marker)
            if order is None:
                continue
            
            # Apply vowel to consonant
            new_char = self.apply_vowel_to_consonant(base_char, order)
            result_chars[idx] = new_char
        
        return "".join(result_chars)

    # ============================================================
    # METHODS FOR TOKENIZER PREPARATION
    # ============================================================
    
    def prepare_for_tokenizer(self, text: str) -> str:
        """
        Full preparation pipeline for SentencePiece:
        1. Apply syllable breaking
        2. Replace problematic characters that would cause splitting
        
        Args:
            text: Input text with word boundary markers
            
        Returns:
            Text ready for SentencePiece tokenization
        """
        # First apply syllable breaking to the entire text
        words = text.split(self.WORD_BOUNDARY)
        processed_words = []
        
        for word in words:
            if word:  # Skip empty
                # Apply syllable breaking
                broken = self.syllable_break(word)
                
                # Replace problematic characters
                for old, new in self.problematic_chars.items():
                    broken = broken.replace(old, new)
                
                processed_words.append(broken)
        
        # Rejoin with word boundary markers
        return self.WORD_BOUNDARY.join(processed_words)
    
    def after_tokenization_restore(self, text: str) -> str:
        """
        Restore original characters after tokenization.
        
        Args:
            text: Text from tokenizer with replaced characters
            
        Returns:
            Text with original parentheses and brackets restored
        """
        for old, new in self.reverse_chars.items():
            text = text.replace(old, new)
        return text
    
    def tokenize_pipeline(self, text: str, tokenizer) -> list:
        """
        Complete tokenization pipeline:
        1. Prepare text (syllable break + replace problematic chars)
        2. Tokenize with SentencePiece
        3. Restore original characters
        
        Args:
            text: Original input text
            tokenizer: SentencePiece tokenizer instance
            
        Returns:
            List of tokens with original characters restored
        """
        # Step 1: Prepare for tokenizer
        prepared = self.prepare_for_tokenizer(text)
        
        # Step 2: Tokenize
        tokens = tokenizer.tokenize(prepared)
        
        # Step 3: Restore original characters in each token
        restored_tokens = [self.after_tokenization_restore(token) for token in tokens]
        
        return restored_tokens

    # ============================================================
    # UTILITY METHODS
    # ============================================================
    
    def get_consonant_skeleton(self, word: str) -> str:
        """
        Extract consonant skeleton from a word.
        
        Args:
            word: Original Ge'ez word
            
        Returns:
            Consonant skeleton (base consonants without vowels)
        """
        broken = self.syllable_break(word)
        skeleton = []
        i = 0
        while i < len(broken):
            if broken[i] == self.WORD_BOUNDARY:
                skeleton.append(broken[i])
                i += 1
            elif self.is_letter_in_language(broken[i]):
                skeleton.append(broken[i])
                # Skip vowel marker if present
                if i + 1 < len(broken) and broken[i+1].startswith('['):
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        return ''.join(skeleton)

    def get_vowel_tags(self, word: str, encode_map: dict) -> str:
        """
        Get vowel tags for a word using encode map.
        
        Args:
            word: Original Ge'ez word
            encode_map: Maps reduction keys to tags
            
        Returns:
            String of tags (e.g., "⟨1⟩⟨2⟩⟨3⟩")
        """
        broken = self.syllable_break(word)
        tags = []
        cons_idx = 0
        i = 0
        while i < len(broken):
            if broken[i] == self.WORD_BOUNDARY:
                i += 1
            elif self.is_letter_in_language(broken[i]):
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
        return ''.join(tags)

    def round_trip_test(self, word: str, encode_map: dict, decode_map: dict) -> bool:
        """
        Test if a word can be encoded and decoded successfully.
        
        Args:
            word: Original Ge'ez word
            encode_map: Maps reduction keys to tags
            decode_map: Maps tags to reduction keys
            
        Returns:
            True if round-trip successful, False otherwise
        """
        # Encode
        skeleton = self.get_consonant_skeleton(word)
        tags_str = self.get_vowel_tags(word, encode_map)
        
        # Extract individual tags
        tags = re.findall(r'⟨\d+⟩', tags_str)
        
        # Decode
        decoded = self.decode_from_splinter(skeleton, tags, decode_map)
        
        success = decoded == word
        if not success:
            print(f"Round-trip failed: '{word}' → '{decoded}'")
        
        return success

    def demonstrate_fix(self):
        """Show the fixed pipeline with examples."""
        examples = [
            "1(:ኦሪት",
            "ዘደገመ⟨15⟩⟨4⟩⟨11⟩",
            ")ዝንቱ",
            "▁ [ ስ ሞ",
            "ቋንቋ",  # Labiovelar example
            "ኰከበ",  # Another labiovelar example
        ]
        
        print("\n" + "="*70)
        print("FIXED PIPELINE DEMONSTRATION")
        print("="*70)
        
        for text in examples:
            print(f"\nOriginal:      '{text}'")
            
            # Step 1: Syllable break
            broken = self.syllable_break(text)
            print(f"After syllable_break: '{broken}'")
            
            # Step 2: Prepare for tokenizer
            prepared = self.prepare_for_tokenizer(text)
            print(f"Prepared for tokenizer: '{prepared}'")
            
            # Show what SentencePiece would see
            print(f"  (Problematic chars replaced: ( → ｟, ) → ｠, [ → ｢, ] → ｣)")
            
            # Step 3: Show restoration (simulated)
            restored = self.after_tokenization_restore(prepared)
            print(f"After restoration: '{restored}'")
        
        print("="*70)


# For quick testing
if __name__ == "__main__":
    utils = EthiopicUtils()
    utils.demonstrate_fix()