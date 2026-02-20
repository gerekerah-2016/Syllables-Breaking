"""
Decode tokenized corpora back to original text.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import json
import re
from pathlib import Path
from src.logger import get_logger
from src.language_utils.EthiopicUtils import EthiopicUtils
from src.utils.path_utils import get_splinter_dir, get_tokenized_corpus_path, get_decoded_corpus_path


class CorpusDecoder:
    """
    Decodes tokenized corpora back to original text.
    """
    
    def __init__(self, language_utils):
        self.language_utils = language_utils
        self.decode_map = None
        self.logger = get_logger()
    
    def load_decode_map(self):
        """Load the decode map from splinter directory."""
        splinter_dir = get_splinter_dir()
        decode_map_path = splinter_dir / "new_unicode_chars_inverted.json"
        
        if not decode_map_path.exists():
            self.logger.error(f"Decode map not found: {decode_map_path}")
            return False
        
        with open(decode_map_path, 'r', encoding='utf-8') as f:
            self.decode_map = json.load(f)
        
        self.logger.info(f"Loaded decode map with {len(self.decode_map)} mappings")
        return True
    
    def decode_word(self, word: str) -> str:
        """
        Decode a single word from its encoded form.
        
        Args:
            word: Encoded word (e.g., "ነገረ⟨4⟩" or "አረተ⟨11⟩⟨18⟩⟨4⟩")
            
        Returns:
            Decoded word with vowels restored
        """
        # Extract consonant skeleton (all Ethiopic letters)
        skeleton = ''.join([c for c in word if self.language_utils.is_letter_in_language(c)])
        
        # Extract tags
        tags = re.findall(r'⟨\d+⟩', word)
        
        if not tags:
            return word
        
        # Decode
        try:
            return self.language_utils.decode_from_splinter(skeleton, tags, self.decode_map)
        except Exception as e:
            self.logger.warning(f"Failed to decode '{word}': {e}")
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
        
        # Split into words (preserve word boundaries)
        words = line.strip().split()
        decoded_words = []
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Check if this word has tags attached or following
            if word.startswith('⟨') and word.endswith('⟩'):
                # This is a standalone tag - should be attached to previous word
                if decoded_words:
                    decoded_words[-1] += word
                else:
                    # Rare case: line starts with a tag
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
    
    def decode_corpus(self, input_path: Path, output_path: Path):
        """
        Decode an entire tokenized corpus file.
        
        Args:
            input_path: Path to tokenized corpus
            output_path: Path to save decoded corpus
        """
        self.logger.info(f"Decoding {input_path} -> {output_path}")
        
        line_count = 0
        word_count = 0
        tagged_word_count = 0
        
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            for line in fin:
                line = line.rstrip('\n')
                
                # Count words in original line
                original_words = line.split()
                word_count += len(original_words)
                
                # Count words with tags
                for word in original_words:
                    if re.search(r'⟨\d+⟩', word):
                        tagged_word_count += 1
                
                # Decode the line
                decoded_line = self.decode_line(line)
                fout.write(decoded_line + '\n')
                
                line_count += 1
                if line_count % 1000 == 0:
                    self.logger.info(f"  Decoded {line_count} lines")
        
        self.logger.info(f"✓ Decoded {line_count} lines")
        self.logger.info(f"  Words processed: {word_count}")
        self.logger.info(f"  Words with tags: {tagged_word_count} ({tagged_word_count/word_count*100:.1f}%)")
    
    def decode_all_tokenized_corpora(self, tokenizer_types=None, vocab_sizes=None):
        """
        Decode all tokenized corpora in the experiment.
        
        Args:
            tokenizer_types: List of tokenizer types (e.g., ['bpe', 'unigram'])
            vocab_sizes: List of vocabulary sizes
        """
        if not self.load_decode_map():
            return
        
        if tokenizer_types is None:
            tokenizer_types = ['bpe', 'unigram']
        
        if vocab_sizes is None:
            vocab_sizes = [10000]  # Default to your trained size
        
        for tokenizer_type in tokenizer_types:
            for vocab_size in vocab_sizes:
                # Check if tokenized corpus exists
                input_path = get_tokenized_corpus_path("splintered", tokenizer_type, vocab_size)
                
                if not input_path.exists():
                    self.logger.warning(f"Tokenized corpus not found: {input_path}")
                    continue
                
                # Decode it
                output_path = get_decoded_corpus_path("splintered", tokenizer_type, vocab_size)
                self.decode_corpus(input_path, output_path)


# For standalone use
if __name__ == "__main__":
    from src.language_utils.EthiopicUtils import EthiopicUtils
    
    utils = EthiopicUtils()
    decoder = CorpusDecoder(utils)
    decoder.decode_all_tokenized_corpora()