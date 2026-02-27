"""
Standalone decoder - no dependencies on experiment parameters or path utils
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import re
import json
from pathlib import Path


class SplinterDecoderStandalone:
    """
    Standalone decoder that doesn't depend on experiment parameters.
    """
    
    def __init__(self, language_utils):
        self.language_utils = language_utils
        self.decode_map = None
        self.cjk_range = (0x4E00, 0x9FFF)
        self.decode_cache = {}
    
    def load_decode_map(self, map_path=None):
        """Load decode map from file."""
        if map_path is None:
            # Look for the map in common locations
            possible_paths = [
                Path("../outputs/2026-02-17-geez-all_letters/results/splinter/new_unicode_chars_inverted.json"),
                Path("outputs/2026-02-17-geez-all_letters/results/splinter/new_unicode_chars_inverted.json"),
                Path("./new_unicode_chars_inverted.json"),
                Path("../new_unicode_chars_inverted.json"),
            ]
            
            for path in possible_paths:
                if path.exists():
                    map_path = path
                    print(f"Found decode map at: {map_path}")
                    break
        
        if not map_path or not Path(map_path).exists():
            print("Decode map not found. Please provide path.")
            return False
        
        with open(map_path, 'r', encoding='utf-8') as f:
            self.decode_map = json.load(f)
        
        print(f"Loaded {len(self.decode_map)} mappings")
        return True
    
    def _is_cjk_char(self, char):
        """Check if character is in CJK Unified Ideographs range."""
        if len(char) != 1:
            return False
        return 0x4E00 <= ord(char) <= 0x9FFF
    
    def _reconstruct_word(self, token):
        """Reconstruct a single word."""
        if token in self.decode_cache:
            return self.decode_cache[token]
        
        # Split into Ge'ez and CJK
        skeleton_chars = []
        cjk_chars = []
        
        for char in token:
            if self.language_utils.is_letter_in_language(char):
                skeleton_chars.append(char)
            elif self._is_cjk_char(char):
                cjk_chars.append(char)
            else:
                skeleton_chars.append(char)
        
        skeleton = ''.join(skeleton_chars)
        
        if not cjk_chars or not self.decode_map:
            return skeleton
        
        # Convert CJK to reduction keys
        reduction_keys = []
        for cjk in cjk_chars:
            if cjk in self.decode_map:
                reduction_keys.append(self.decode_map[cjk])
        
        if not reduction_keys:
            return skeleton
        
        # Apply vowels
        if hasattr(self.language_utils, 'decode_from_splinter_with_keys'):
            result = self.language_utils.decode_from_splinter_with_keys(
                skeleton, reduction_keys, self.decode_map
            )
        else:
            result = skeleton
        
        self.decode_cache[token] = result
        return result
    
    def decode_line(self, line):
        """Decode a line."""
        if not line:
            return line
        
        # Remove SentencePiece markers
        line = line.replace('▁', '')
        
        # Split into tokens
        tokens = line.strip().split()
        reconstructed = []
        
        i = 0
        while i < len(tokens):
            current = tokens[i]
            has_cjk = any(self._is_cjk_char(c) for c in current)
            
            if has_cjk:
                # Combined format
                word = self._reconstruct_word(current)
                reconstructed.append(word)
                i += 1
            else:
                # Maybe separate format
                if (i + 1 < len(tokens) and 
                    any(self._is_cjk_char(c) for c in tokens[i+1])):
                    combined = current + tokens[i+1]
                    word = self._reconstruct_word(combined)
                    reconstructed.append(word)
                    i += 2
                else:
                    reconstructed.append(current)
                    i += 1
        
        result = ' '.join(reconstructed)
        
        # Fix punctuation
        result = re.sub(r'\s+([፡።፣፤፥፦፧፠፨])', r'\1', result)
        result = re.sub(r'\s+', ' ', result)
        
        return result.strip()