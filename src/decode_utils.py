"""
Decoding utilities with Rényi efficiency calculation - FIXED to remove tag artifacts
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import re
import json
import math
import numpy as np
from collections import Counter
from pathlib import Path
from src.logger import get_logger
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface
from src.utils.path_utils import get_splinter_dir


class SplinterDecoder:
    """
    Decodes tokenized text back to original Ge'ez.
    PRESERVES all punctuation and structure.
    REMOVES all tag artifacts completely.
    Includes Rényi efficiency calculation.
    """
    
    def __init__(self, language_utils: LanguageUtilsInterface):
        self.language_utils = language_utils
        self.decode_map = None
        self.logger = get_logger()
    
    def load_decode_map(self):
        """Load decode map from file."""
        splinter_dir = get_splinter_dir()
        map_path = splinter_dir / "new_unicode_chars_inverted.json"
        
        if not map_path.exists():
            self.logger.error(f"Decode map not found: {map_path}")
            return False
        
        with open(map_path, 'r', encoding='utf-8') as f:
            self.decode_map = json.load(f)
        
        self.logger.info(f"Loaded {len(self.decode_map)} mappings")
        return True
    
    def decode_line(self, line: str) -> str:
        """
        Decode a single line from tokenized file.
        GEMINI'S FIXED VERSION + punctuation fixes.
        """
        if not line or not line.strip():
            return line
        
        # Step 1: Fix tokenizer artifacts (spaces inside tags)
        line = re.sub(r'⟨\s+(\d+)\s+⟩', r'⟨\1⟩', line)
        
        # Step 2: Remove SentencePiece ▁ markers
        line = line.replace('▁', '')
        
        # Step 3: Split into tokens (words/punctuation)
        tokens = line.strip().split()
        decoded_parts = []
        
        for token in tokens:
            # Step 4: Extract ONLY complete tags with both brackets ⟨n⟩
            # This is SAFE - won't match regular numbers like "2016"
            tags = re.findall(r'⟨\d+⟩', token)
            
            # Step 5: Extract Ge'ez letters for skeleton
            skeleton = ''.join([c for c in token 
                               if self.language_utils.is_letter_in_language(c)])
            
            if tags and skeleton and self.decode_map:
                # Step 6: Apply SPLINTER decoding (አረተ + ⟨27⟩ → ኦሪት)
                decoded_word = self.language_utils.decode_from_splinter(
                    skeleton, tags, self.decode_map
                )
                
                # Step 7: Extract ONLY punctuation (remove Ge'ez letters and complete tags)
                # This preserves real numbers and punctuation
                punctuation_only = re.sub(r'[ሀ-ፐ]|⟨\d+⟩', '', token)
                
                # Step 8: Combine decoded word with its punctuation
                decoded_parts.append(decoded_word + punctuation_only)
            else:
                # Step 9: No tags - just remove any stray tag artifacts
                clean_token = re.sub(r'⟨\d+⟩', '', token)
                decoded_parts.append(clean_token)
        
        # Step 10: Join and clean up spacing around punctuation
        result = ' '.join(decoded_parts)
        
        # Step 11: YOUR PUNCTUATION FIXES (added back)
        # Fix spaces before punctuation
        result = re.sub(r'\s+([፡።፣፤፥፦፧፠፨\(\)\[\]\{\}.,;:!?])', r'\1', result)
        # Fix spaces after punctuation
        result = re.sub(r'([፡።፣፤፥፦፧፠፨\(\)\[\]\{\}.,;:!?])\s+', r'\1', result)
        
        # Fix numbers with parentheses
        result = re.sub(r'(\d+)\s+\(\s+:', r'\1(:', result)
        result = re.sub(r'\(\s+:', r'(:', result)
        
        # Fix punctuation combinations
        result = re.sub(r'([።፡፣፤])\s+\)', r'\1)', result)
        result = re.sub(r'\(\s+([።፡፣፤])', r'(\1', result)
        
        return result.strip()
    
    def decode_file(self, input_path: Path, output_path: Path):
        """Decode entire file."""
        self.logger.info(f"Decoding: {input_path}")
        
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
                    self.logger.info(f"  Processed {line_count} lines")
        
        self.logger.info(f"Complete: {line_count} lines")
        return line_count
    
    def compute_renyi_efficiency(self, tokenized_path: Path, alpha: float = 2.0, sample_lines: int = 10000):
        """
        Compute Rényi efficiency for tokenized corpus.
        """
        self.logger.info(f"Computing Rényi efficiency (α={alpha}) for {tokenized_path}")
        
        # Count token frequencies
        token_counter = Counter()
        total_tokens = 0
        line_count = 0
        
        with open(tokenized_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line_count >= sample_lines:
                    break
                tokens = line.strip().split()
                token_counter.update(tokens)
                total_tokens += len(tokens)
                line_count += 1
        
        if total_tokens == 0:
            return {"error": "No tokens found"}
        
        # Calculate probabilities
        probs = np.array(list(token_counter.values())) / total_tokens
        
        # Calculate Rényi entropy
        if alpha == 1.0:
            renyi_entropy = -np.sum(probs * np.log(probs + 1e-10))
        else:
            renyi_entropy = (1 / (1 - alpha)) * np.log(np.sum(probs ** alpha) + 1e-10)
        
        # Calculate efficiency
        n = len(probs)
        if n <= 1:
            efficiency = 1.0
        else:
            efficiency = np.exp(renyi_entropy) / n
        
        results = {
            "alpha": alpha,
            "total_tokens": total_tokens,
            "unique_tokens": n,
            "renyi_entropy": float(renyi_entropy),
            "efficiency": float(efficiency),
            "sample_lines": line_count
        }
        
        self.logger.info(f"  Unique tokens: {n}")
        self.logger.info(f"  Rényi entropy: {renyi_entropy:.4f}")
        self.logger.info(f"  Efficiency: {efficiency:.4f}")
        
        return results
    
    def evaluate_round_trip(self, original_path: Path, tokenized_path: Path, sample_lines: int = 1000):
        """
        Evaluate round-trip accuracy.
        """
        self.logger.info(f"Evaluating round-trip accuracy")
        
        # First decode the tokenized file to a temporary location
        temp_path = tokenized_path.parent / f"temp_decoded_{tokenized_path.stem}.txt"
        self.decode_file(tokenized_path, temp_path)
        
        # Compare original vs decoded
        correct = 0
        total = 0
        
        with open(original_path, 'r', encoding='utf-8') as f_orig, \
             open(temp_path, 'r', encoding='utf-8') as f_dec:
            
            for i, (orig_line, dec_line) in enumerate(zip(f_orig, f_dec)):
                if i >= sample_lines:
                    break
                
                orig = orig_line.strip()
                dec = dec_line.strip()
                
                if orig and dec:
                    if orig == dec:
                        correct += 1
                    total += 1
        
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()
        
        accuracy = correct / total if total > 0 else 0
        
        results = {
            "total_samples": total,
            "correct": correct,
            "accuracy": float(accuracy)
        }
        
        self.logger.info(f"  Round-trip accuracy: {accuracy:.2%} ({correct}/{total})")
        
        return results
    
    def full_evaluation(self, original_path: Path, tokenized_path: Path, output_path: Path = None):
        """
        Run full evaluation including Rényi efficiency and round-trip accuracy.
        """
        self.logger.info("=" * 60)
        self.logger.info("FULL EVALUATION")
        self.logger.info("=" * 60)
        
        results = {}
        
        # Rényi efficiency for different alpha values
        for alpha in [0.5, 1.0, 2.0, 3.0]:
            results[f"renyi_alpha_{alpha}"] = self.compute_renyi_efficiency(
                tokenized_path, alpha=alpha
            )
        
        # Round-trip accuracy
        results["round_trip"] = self.evaluate_round_trip(original_path, tokenized_path)
        
        # Save results if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Results saved to {output_path}")
        
        return results