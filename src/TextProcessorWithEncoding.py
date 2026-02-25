"""
Text processor with encoding - FINAL VERSION with proper ፡ splitting
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import re
import json
from src.TextProcessorInterface import TextProcessorInterface
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface


class TextProcessorWithEncoding(TextProcessorInterface):
    """
    Encodes Ethiopic text using ⟨n⟩ tags.
    PRESERVES all numbers, punctuation, spaces, and line breaks.
    CRITICAL: Splits on ፡ and preserves it as a separator!
    """
    
    def __init__(self, language_utils: LanguageUtilsInterface, 
                 reductions_map, new_unicode_chars_map, 
                 new_unicode_chars_inverted_map=None):
        super().__init__(language_utils)
        self.reductions_map = reductions_map
        self.encode_map = new_unicode_chars_map
        self.decode_map = new_unicode_chars_inverted_map
        self.next_tag = self._get_next_tag_number()
        self.word_cache = {}
        
        # Ethiopic punctuation that MUST be preserved
        self.ETHIOPIC_PUNCTUATION = {'፡', '።', '፣', '፤', '፥', '፦', '፧', '፠', '፨'}
        
    def _get_next_tag_number(self):
        if not self.decode_map:
            return 1
        
        max_num = 0
        for tag in self.decode_map.keys():
            if tag.startswith('⟨') and tag.endswith('⟩'):
                try:
                    num = int(tag[1:-1])
                    max_num = max(max_num, num)
                except:
                    pass
        return max_num + 1
    
    def _has_geez(self, text):
        return any(self.language_utils.is_letter_in_language(c) for c in text)
    
    def _clean_text(self, text):
        """Clean text - PRESERVE all characters!"""
        return text  # Return as-is, no cleaning!
    
    def process(self, text):
        """
        Process text with splinter encoding.
        FIXED: Splits on both whitespace AND Ethiopic punctuation.
        """
        if text is None:
            return ''
        
        # Process line by line
        lines = text.split('\n')
        encoded_lines = []
        
        for line in lines:
            if not line:
                encoded_lines.append('')
                continue
            
            # CRITICAL: Split on both whitespace AND Ethiopic punctuation
            # Using regex with capturing group to preserve the separators
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
        
        return '\n'.join(encoded_lines)
    
    def encode_word(self, word):
        """
        Encode a single word.
        FIXED: Preserves ALL original characters.
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
        Encode a pure Ge'ez word (no punctuation).
        """
        # Get base consonants and vowel orders
        bases = []
        orders = []
        
        for char in geez_word:
            base, order = self.language_utils.get_base_and_order(char)
            bases.append(base)
            orders.append(order)
        
        # Build skeleton
        skeleton = ''.join(bases)
        
        # Create tags for non-base vowels
        tags = []
        for idx, order in enumerate(orders):
            if order > 0:
                marker = self.language_utils.VOWEL_SYMBOLS.get(order, f"[{order}]")
                key = f"{idx}:{marker}"
                
                if key in self.encode_map:
                    tags.append(self.encode_map[key])
                else:
                    tag = f"⟨{self.next_tag}⟩"
                    self.encode_map[key] = tag
                    if self.decode_map is not None:
                        self.decode_map[tag] = key
                    tags.append(tag)
                    self.next_tag += 1
        
        return skeleton + ''.join(tags)
    
    def decode_word(self, encoded_word):
        """
        Decode an encoded word back to original.
        """
        # Process character by character
        result_parts = []
        current_geez = []
        current_tags = []
        in_tag = False
        tag_buffer = []
        
        for char in encoded_word:
            if char == '⟨':
                if current_geez:
                    result_parts.append(''.join(current_geez))
                    current_geez = []
                in_tag = True
                tag_buffer = [char]
            elif char == '⟩':
                tag_buffer.append(char)
                current_tags.append(''.join(tag_buffer))
                tag_buffer = []
                in_tag = False
            elif in_tag:
                tag_buffer.append(char)
            elif self.language_utils.is_letter_in_language(char):
                current_geez.append(char)
            else:
                if current_geez:
                    result_parts.append(''.join(current_geez))
                    current_geez = []
                result_parts.append(char)
        
        if current_geez:
            result_parts.append(''.join(current_geez))
        
        # If no tags, return as-is
        if not current_tags or not self.decode_map:
            return ''.join(result_parts)
        
        # Decode the Ge'ez part
        skeleton = ''.join([c for c in encoded_word 
                           if self.language_utils.is_letter_in_language(c)])
        
        decoded_geez = self.language_utils.decode_from_splinter(
            skeleton, current_tags, self.decode_map
        )
        
        # Replace the skeleton in the result with decoded Ge'ez
        final_result = []
        geez_idx = 0
        
        for part in result_parts:
            if all(self.language_utils.is_letter_in_language(c) for c in part):
                # This is a Ge'ez part
                if geez_idx < len(decoded_geez):
                    final_result.append(decoded_geez[geez_idx:geez_idx+len(part)])
                    geez_idx += len(part)
                else:
                    final_result.append(part)
            else:
                final_result.append(part)
        
        return ''.join(final_result)
    
    def save_maps(self, encode_path, decode_path):
        with open(encode_path, 'w', encoding='utf-8') as f:
            json.dump(self.encode_map, f, indent=2, ensure_ascii=False)
        
        if self.decode_map:
            with open(decode_path, 'w', encoding='utf-8') as f:
                json.dump(self.decode_map, f, indent=2, ensure_ascii=False)