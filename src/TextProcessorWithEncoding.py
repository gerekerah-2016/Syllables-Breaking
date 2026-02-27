"""
Text processor with encoding - PURE CJK APPROACH
No brackets, no tags - direct CJK character mapping.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import re
import json
from src.TextProcessorInterface import TextProcessorInterface
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface
from src.logger import get_logger


class TextProcessorWithEncoding(TextProcessorInterface):
    """
    Encodes Ethiopic text using direct CJK character mapping.
    This achieves 0.5-0.7 Rényi efficiency by reducing sequence length.
    PRESERVES all numbers, punctuation, spaces, and line breaks.
    CRITICAL: Splits on ፡ and preserves it as a separator!
    """
    
    def __init__(self, language_utils: LanguageUtilsInterface, 
                 reductions_map, new_unicode_chars_map, 
                 new_unicode_chars_inverted_map=None):
        super().__init__(language_utils)
        self.reductions_map = reductions_map
        self.encode_map = new_unicode_chars_map  # Maps reduction keys to CJK chars
        self.decode_map = new_unicode_chars_inverted_map  # Maps CJK chars to reduction keys
        self.word_cache = {}
        self.logger = get_logger()
        
        # Ethiopic punctuation that MUST be preserved
        self.ETHIOPIC_PUNCTUATION = {'፡', '።', '፣', '፤', '፥', '፦', '፧', '፠', '፨'}
        
        # CJK range for identification
        self.CJK_RANGE = (0x4E00, 0x9FFF)
        
        # Track missing mappings for debugging
        self.missing_mappings_count = 0
        self.missing_mappings_logged = set()
        self._summary_printed = False  # Flag to ensure summary prints only once
        
    def _has_geez(self, text):
        return any(self.language_utils.is_letter_in_language(c) for c in text)
    
    def _is_cjk_char(self, char):
        """Check if character is in CJK Unified Ideographs range."""
        if len(char) != 1:
            return False
        code = ord(char)
        return self.CJK_RANGE[0] <= code <= self.CJK_RANGE[1]
    
    def _filter_unknown_cjk(self, text):
        """
        Remove any CJK characters that aren't in our allowed set.
        This prevents unknown CJK characters from appearing in the output.
        """
        # Get the set of allowed CJK characters from encode_map values
        allowed_cjk = set(self.encode_map.values()) if self.encode_map else set()
        
        result = []
        removed_count = 0
        
        for char in text:
            # Keep Ge'ez letters
            if self.language_utils.is_letter_in_language(char):
                result.append(char)
            # Keep Ethiopic punctuation
            elif char in self.ETHIOPIC_PUNCTUATION:
                result.append(char)
            # Keep whitespace
            elif char.isspace():
                result.append(char)
            # Keep numbers
            elif char.isdigit():
                result.append(char)
            # Keep ASCII punctuation
            elif ord(char) < 128:
                result.append(char)
            # Keep ALLOWED CJK characters (our tags)
            elif char in allowed_cjk:
                result.append(char)
            # Remove ALL OTHER CJK characters
            elif self._is_cjk_char(char):
                removed_count += 1
                # Skip - don't add to result
            # Keep everything else
            else:
                result.append(char)
        
        if removed_count > 0:
            self.logger.debug(f"Filtered out {removed_count} unknown CJK characters")
        
        return ''.join(result)
    
    def _clean_text(self, text):
        """Clean text - PRESERVE all characters, but filter unknown CJK."""
        return self._filter_unknown_cjk(text)
    
    def process(self, text):
        """
        Process text with splinter encoding using direct CJK characters.
        """
        if text is None:
            return ''
        
        # First filter out unknown CJK characters
        text = self._filter_unknown_cjk(text)
        
        # Process line by line
        lines = text.split('\n')
        encoded_lines = []
        
        total_lines = len(lines)
        for idx, line in enumerate(lines):
            if not line:
                encoded_lines.append('')
                continue
            
            # Split on both whitespace AND Ethiopic punctuation
            tokens = re.split(r'(\s+|፡|።|፣|፤|፥|፦|፧|፠|፨)', line)
            
            # Process each token
            encoded_parts = []
            for token in tokens:
                if not token:
                    continue
                
                # If token is whitespace or punctuation, keep it as-is
                if token.isspace() or token in self.ETHIOPIC_PUNCTUATION:
                    encoded_parts.append(token)
                # Otherwise, it's a word - encode it if it has Ge'ez
                elif self._has_geez(token):
                    encoded_parts.append(self.encode_word(token))
                else:
                    encoded_parts.append(token)
            
            encoded_lines.append(''.join(encoded_parts))
            
            # Add progress indicator every 100,000 lines
            if (idx + 1) % 100000 == 0:
                percent = (idx + 1) / total_lines * 100 if total_lines > 0 else 0
                self.logger.info(f"  Processing progress: {idx+1}/{total_lines} lines ({percent:.1f}%)")
        
        # Return encoded lines without printing summary here
        return '\n'.join(encoded_lines)
    
    def print_summary(self):
        """Print summary statistics (call this once after all processing)."""
        if not self._summary_printed:
            if self.missing_mappings_count > 0:
                self.logger.info(f"  Processing complete. Total missing mappings: {self.missing_mappings_count}")
                if self.missing_mappings_logged:
                    self.logger.info(f"  Unique missing mapping types: {len(self.missing_mappings_logged)}")
            else:
                self.logger.info("  Processing complete. No missing mappings encountered.")
            self._summary_printed = True
    
    def encode_word(self, word):
        """
        Encode a single word using direct CJK characters.
        Example: "ኦሪት" → "አረተ二十七一" (where 二十七一 are CJK chars)
        """
        if word in self.word_cache:
            return self.word_cache[word]
        
        # Process character by character
        result_parts = []
        geez_buffer = []
        
        for char in word:
            if self.language_utils.is_letter_in_language(char):
                geez_buffer.append(char)
            else:
                # If we have accumulated Ge'ez letters, encode them first
                if geez_buffer:
                    geez_word = ''.join(geez_buffer)
                    encoded = self._encode_geez_word(geez_word)
                    result_parts.append(encoded)
                    geez_buffer = []
                # Add non-Ge'ez character as-is
                result_parts.append(char)
        
        # Handle any remaining Ge'ez letters
        if geez_buffer:
            geez_word = ''.join(geez_buffer)
            encoded = self._encode_geez_word(geez_word)
            result_parts.append(encoded)
        
        result = ''.join(result_parts)
        self.word_cache[word] = result
        return result
    
    def _encode_geez_word(self, geez_word):
        """
        Encodes pure Ge'ez using direct CJK characters.
        No brackets, no tags - pure CJK mapping.
        """
        # Get base consonants and vowel orders
        bases = []
        orders = []
        
        for char in geez_word:
            base, order = self.language_utils.get_base_and_order(char)
            bases.append(base)
            orders.append(order)
        
        skeleton = ''.join(bases)
        cjk_chars = []
        missing_in_word = 0
        
        for idx, order in enumerate(orders):
            if order > 0:
                # Use standard vowel markers: [u], [i], [a], etc.
                marker = self.language_utils.VOWEL_SYMBOLS.get(order, f"[{order}]")
                key = f"{idx}:{marker}"
                
                # Check if we already have a CJK character for this pos:vowel key
                if key in self.encode_map:
                    cjk_chars.append(self.encode_map[key])
                else:
                    missing_in_word += 1
                    self.missing_mappings_count += 1
                    
                    # Log each unique missing mapping only once at DEBUG level
                    if key not in self.missing_mappings_logged:
                        self.missing_mappings_logged.add(key)
                        self.logger.debug(f"Missing mapping for {key} in word: {geez_word}")
        
        # Only log if many missing in a single word (optional)
        if missing_in_word > 3:
            self.logger.debug(f"Word has {missing_in_word} missing mappings: {geez_word}")
        
        return skeleton + ''.join(cjk_chars)
    
    def decode_word(self, encoded_word):
        """
        Decode a word with CJK characters back to original Ge'ez.
        """
        # Split into Ge'ez segments, CJK tag segments, and "Other"
        # This regex groups consecutive Ge'ez letters and consecutive CJK chars
        parts = re.findall(r'[\u1200-\u137F]+|[\u4E00-\u9FFF]+|[^\u1200-\u137F\u4E00-\u9FFF]+', encoded_word)
        
        final_decoded = []
        last_geez = ""

        for part in parts:
            if not part:
                continue
            
            first_char = part[0]
            
            # 1. If it's a Ge'ez Skeleton
            if self.language_utils.is_letter_in_language(first_char):
                last_geez = part
                # We don't add to final_decoded yet, because tags might follow
            
            # 2. If it's a string of CJK Tags
            elif self._is_cjk_char(first_char):
                if last_geez and self.decode_map:
                    # Part is a string of CJK chars like '一丁'
                    # Convert each CJK char to its reduction key
                    reduction_keys = [self.decode_map.get(c) for c in part if c in self.decode_map]
                    # Remove None values
                    reduction_keys = [k for k in reduction_keys if k is not None]
                    
                    if reduction_keys:
                        # Use decode_from_splinter_with_keys
                        if hasattr(self.language_utils, 'decode_from_splinter_with_keys'):
                            decoded = self.language_utils.decode_from_splinter_with_keys(
                                last_geez, reduction_keys, self.decode_map
                            )
                        else:
                            # Fallback
                            decoded = last_geez
                        final_decoded.append(decoded)
                    else:
                        # If no valid keys, just keep the skeleton
                        final_decoded.append(last_geez)
                    last_geez = ""  # Reset
                else:
                    # Orphan CJK chars - shouldn't happen in clean data
                    final_decoded.append(part)
            
            # 3. Everything else (Punctuation/ASCII)
            else:
                if last_geez:
                    final_decoded.append(last_geez)
                    last_geez = ""
                final_decoded.append(part)

        # Add any remaining Ge'ez
        if last_geez:
            final_decoded.append(last_geez)

        return ''.join(final_decoded)
    
    def save_maps(self, encode_path, decode_path):
        """Save maps to files."""
        with open(encode_path, 'w', encoding='utf-8') as f:
            json.dump(self.encode_map, f, indent=2, ensure_ascii=False)
        
        if self.decode_map:
            with open(decode_path, 'w', encoding='utf-8') as f:
                json.dump(self.decode_map, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved encode map ({len(self.encode_map)} entries) to {encode_path}")
        if self.decode_map:
            self.logger.info(f"Saved decode map ({len(self.decode_map)} entries) to {decode_path}")