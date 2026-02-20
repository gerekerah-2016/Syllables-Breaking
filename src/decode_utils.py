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
    """
    
    def __init__(self, language_utils: LanguageUtilsInterface):
        self.language_utils = language_utils
        self.decode_map = None
        self.logger = get_logger()
    
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
    
    def decode_word(self, word: str) -> str:
        """
        Decode a single word from its encoded form.
        
        Args:
            word: Encoded word (e.g., "አረተ⟨11⟩⟨18⟩⟨4⟩")
            
        Returns:
            Decoded word with vowels restored
        """
        # Extract consonant skeleton (all Ethiopic letters)
        skeleton = ''.join([c for c in word if self.language_utils.is_letter_in_language(c)])
        
        # Extract tags
        tags = re.findall(r'⟨\d+⟩', word)
        
        if not tags:
            return word
        
        # Decode using language utils
        try:
            decoded = self.language_utils.decode_from_splinter(skeleton, tags, self.decode_map)
            return decoded
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to decode '{word}': {e}")
            return word
    
    def decode_line(self, line: str) -> str:
        """
        Decode an entire line of tokenized text.
        
        Args:
            line: Tokenized line with words and tags
            
        Returns:
            Decoded line with vowels restored
        """
        if not line.strip():
            return line
        
        # Split into words
        words = line.strip().split()
        decoded_words = []
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Check if this is a standalone tag (like punctuation)
            if word.startswith('⟨') and word.endswith('⟩') and len(word) <= 5:
                # This is likely a punctuation tag - keep as is
                decoded_words.append(word)
                i += 1
                continue
            
            # Look ahead for tags that belong to this word
            full_word = word
            while i + 1 < len(words) and words[i + 1].startswith('⟨'):
                full_word += words[i + 1]
                i += 1
            
            # Decode the full word
            decoded_words.append(self.decode_word(full_word))
            i += 1
        
        return ' '.join(decoded_words)
    
    def decode_file(self, input_path: Path, output_path: Path):
        """
        Decode an entire tokenized file.
        
        Args:
            input_path: Path to tokenized file
            output_path: Path to save decoded file
        """
        self.logger.info(f"📥 Decoding: {input_path}")
        self.logger.info(f"📤 Output: {output_path}")
        
        line_count = 0
        word_count = 0
        decoded_word_count = 0
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                line = line.rstrip('\n')
                
                # Count words in original line
                original_words = line.split()
                word_count += len(original_words)
                
                # Count words that will be decoded (have tags)
                for word in original_words:
                    if re.search(r'⟨\d+⟩', word):
                        decoded_word_count += 1
                
                # Decode the line
                decoded_line = self.decode_line(line)
                fout.write(decoded_line + '\n')
                
                line_count += 1
                if line_count % 1000 == 0:
                    self.logger.info(f"  Processed {line_count} lines...")
        
        self.logger.info(f"✓ Complete: {line_count} lines, {word_count} words")
        self.logger.info(f"  Words decoded: {decoded_word_count}")
        return line_count, word_count, decoded_word_count
    
    def verify_decoding(self, original_sample, tokenized_sample, decoded_sample):
        """
        Verify that decoding works by comparing samples.
        
        Args:
            original_sample: List of original lines
            tokenized_sample: List of tokenized lines
            decoded_sample: List of decoded lines
        """
        self.logger.info("🔍 Verifying decoding quality...")
        
        matches = 0
        total = min(len(original_sample), len(decoded_sample))
        
        for i in range(total):
            if original_sample[i].strip() == decoded_sample[i].strip():
                matches += 1
        
        accuracy = (matches / total * 100) if total > 0 else 0
        self.logger.info(f"  Match rate: {matches}/{total} ({accuracy:.1f}%)")
        
        if accuracy > 95:
            self.logger.info("✅ Decoding verification PASSED!")
        else:
            self.logger.warning("⚠️ Decoding verification shows discrepancies")
        
        return accuracy