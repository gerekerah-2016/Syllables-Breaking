"""
Tokenize corpus using trained tokenizers.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import sentencepiece as spm
from src.logger import get_logger


class CorpusTokenizer:
    """Tokenize corpora using trained tokenizers."""
    
    def __init__(self):
        self.logger = get_logger()
    
    def tokenize_corpus_into_file(self, tokenizer_path: str, input_path: str, output_path: str):
        """
        Tokenize a corpus and save to file.
        
        Args:
            tokenizer_path: Path to tokenizer model (without extension)
            input_path: Path to input corpus
            output_path: Path to output tokenized corpus
        """
        self.logger.info(f"Tokenizing {input_path} -> {output_path}")
        
        try:
            # Load tokenizer
            sp = spm.SentencePieceProcessor()
            sp.load(str(tokenizer_path) + ".model")
            
            # Tokenize corpus
            with open(input_path, 'r', encoding='utf-8') as f_in:
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    for i, line in enumerate(f_in):
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Tokenize
                        pieces = sp.encode_as_pieces(line)
                        f_out.write(' '.join(pieces) + '\n')
                        
                        if (i + 1) % 10000 == 0:
                            self.logger.info(f"  Tokenized {i+1} lines")
            
            self.logger.info(f"✓ Tokenized corpus saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to tokenize corpus: {e}")