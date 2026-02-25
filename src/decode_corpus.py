"""
Decoding utilities for SPLINTER tokenized output.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import json
import re
from pathlib import Path
from src.logger import get_logger
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface
from src.utils.path_utils import get_splinter_dir


class SplinterDecoder:
    """
    Decodes SPLINTER-encoded text back to original Ge'ez.
    Handles ⟨n⟩, [Tn], _Tn, and Mathematical Symbol formats.
    """
    
    def __init__(self, language_utils: LanguageUtilsInterface):
        self.language_utils = language_utils
        self.decode_map = None
        self.logger = get_logger()
        self.symbol_to_tag = {}
        self.tag_to_symbol = {}
        self.symbol_pattern = None
        
        # Load symbol mapping
        self._load_symbol_mapping()
    
    def _load_symbol_mapping(self):
        """Load symbol mapping."""
        try:
            possible_paths = [
                Path("tag_mapping.json"),
                Path(__file__).parent / "tag_mapping.json",
                Path(__file__).parent.parent / "tag_mapping.json",
                Path("src/tag_mapping.json")
            ]
            
            mapping_path = None
            for path in possible_paths:
                if path.exists():
                    mapping_path = path
                    break
            
            if mapping_path:
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)
                    self.symbol_to_tag = mapping_data['symbol_to_tag']
                    self.tag_to_symbol = {v: k for k, v in self.symbol_to_tag.items()}
                
                # Create symbol pattern
                if self.symbol_to_tag:
                    symbols = ''.join(re.escape(sym) for sym in self.symbol_to_tag.keys())
                    self.symbol_pattern = re.compile(f'[{symbols}]')
                
                self.logger.info(f"✅ Loaded {len(self.symbol_to_tag)} symbol mappings")
            else:
                self.logger.warning("⚠️ Tag mapping file not found")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load symbol mapping: {e}")
    
    def load_decode_map(self):
        """Load the decode map from splinter directory."""
        splinter_dir = get_splinter_dir()
        decode_map_path = splinter_dir / "new_unicode_chars_inverted.json"
        
        if not decode_map_path.exists():
            self.logger.error(f"❌ Decode map not found: {decode_map_path}")
            return False
        
        with open(decode_map_path, 'r', encoding='utf-8') as f:
            self.decode_map = json.load(f)
        
        self.logger.info(f"✓ Loaded decode map with {len(self.decode_map)} mappings")
        return True
    
    def _is_mathematical_symbol(self, char):
        """Check if character is in Mathematical Alphanumeric range."""
        return 0x1D400 <= ord(char) <= 0x1D7FF
    
    def _is_pua_symbol(self, char):
        """Check if character is in Private Use Area."""
        return 0xE000 <= ord(char) <= 0xF8FF
    
    def _is_tag_symbol(self, char):
        """Check if character is a tag symbol."""
        return self._is_mathematical_symbol(char) or self._is_pua_symbol(char)
    
    def _symbols_to_tags(self, text: str) -> str:
        """Convert symbols to ⟨n⟩ format."""
        if not self.symbol_pattern or not self.symbol_to_tag:
            return text
        
        def replace(match):
            sym = match.group(0)
            tag_num = self.symbol_to_tag.get(sym)
            return f"⟨{tag_num}⟩" if tag_num else sym
        
        return self.symbol_pattern.sub(replace, text)
    
    def decode_word(self, word: str) -> str:
        """
        Decode a single word.
        """
        # Convert any symbols to ⟨n⟩ format
        normalized = self._symbols_to_tags(word)
        
        # Extract skeleton (all Ethiopic letters)
        skeleton = ''.join([c for c in normalized if self.language_utils.is_letter_in_language(c)])
        
        if not skeleton:
            return word
        
        # Extract tags
        tags = re.findall(r'⟨\d+⟩', normalized)
        
        if not tags:
            return skeleton
        
        # Apply tags to transform skeleton
        result = list(skeleton)
        skeleton_length = len(result)
        
        for tag in tags:
            if tag not in self.decode_map:
                self.logger.debug(f"Tag {tag} not found")
                continue
            
            key = self.decode_map[tag]
            if ':' not in key:
                continue
            
            pos_str, vowel_marker = key.split(':', 1)
            try:
                pos = int(pos_str)
            except ValueError:
                continue
            
            if pos >= skeleton_length:
                self.logger.debug(f"Position {pos} out of range, skipping")
                continue
            
            base_char = result[pos]
            order = self.language_utils.get_vowel_order_from_marker(vowel_marker)
            if order is None:
                continue
            
            new_char = self.language_utils.apply_vowel_to_consonant(base_char, order)
            result[pos] = new_char
        
        return ''.join(result)
    
    def decode_line(self, line: str) -> str:
        """Decode an entire line."""
        if not line or not line.strip():
            return line
        
        # Split into tokens
        tokens = line.strip().split()
        
        # Group tokens into words (handling ▁ markers)
        words = []
        current = []
        
        for token in tokens:
            if token.startswith('▁'):
                if current:
                    words.append(''.join(current))
                current = [token[1:]] if len(token) > 1 else []
            else:
                current.append(token)
        
        if current:
            words.append(''.join(current))
        
        # Decode each word
        decoded = []
        for word in words:
            if word:
                decoded.append(self.decode_word(word))
        
        return ' '.join(decoded)
    
    def decode_file(self, input_path: Path, output_path: Path):
        """Decode an entire file."""
        self.logger.info(f"📥 Decoding: {input_path}")
        
        line_count = 0
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                line = line.rstrip('\n')
                if line:
                    decoded = self.decode_line(line)
                    fout.write(decoded + '\n')
                else:
                    fout.write('\n')
                
                line_count += 1
                if line_count % 1000 == 0:
                    self.logger.info(f"  Processed {line_count} lines...")
        
        self.logger.info(f"✓ Complete: {line_count} lines")
        return line_count