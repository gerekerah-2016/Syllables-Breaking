"""
Decoding utilities - FINAL WORKING VERSION
Handles all formats: attached, space-separated, with punctuation
Includes Ghost Index mapping for perfect CJK tag application
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
    Handles both attached and space-separated formats.
    Preserves all punctuation.
    Uses Ghost Index mapping for perfect CJK tag application.
    """
    
    def __init__(self, language_utils: LanguageUtilsInterface):
        self.language_utils = language_utils
        self.decode_map = None
        self.logger = get_logger()
        self.cjk_range = (0x4E00, 0x9FFF)
        self.decode_cache = {}
    
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
    
    def _is_cjk_char(self, char):
        """Check if character is in CJK Unified Ideographs range."""
        if len(char) != 1:
            return False
        return 0x4E00 <= ord(char) <= 0x9FFF
    
    def _reconstruct_word(self, combined_string):
        """
        Reconstruct word using Ghost Index mapping.
        Maps CJK instructions to Ge'ez letters only, ignoring punctuation.
        """
        cache_key = combined_string
        if cache_key in self.decode_cache:
            return self.decode_cache[cache_key]
        
        # Convert to list for modification
        result = list(combined_string)
        
        # 🎯 STEP 1: Find indices of actual Ge'ez letters only
        geez_indices = [
            i for i, char in enumerate(result)
            if self.language_utils.is_letter_in_language(char)
        ]
        
        # Collect all CJK chars and their instructions
        cjk_chars = [c for c in combined_string if self._is_cjk_char(c)]
        
        # 🎯 STEP 2: Parse all instructions from CJK tags
        instructions = []  # (linguistic_position, marker)
        for cjk in cjk_chars:
            if cjk in self.decode_map:
                reduction_key = self.decode_map[cjk]
                parts = reduction_key.split(',')
                for part in parts:
                    part = part.strip()
                    if ':' in part:
                        try:
                            pos_str, marker = part.split(':', 1)
                            instructions.append((int(pos_str), marker.strip()))
                        except ValueError:
                            continue
        
        # Sort instructions by linguistic position
        instructions.sort(key=lambda x: x[0])
        
        # 🎯 STEP 3: Apply vowels using the mapped indices
        modified_geez_positions = set()
        
        for ling_pos, marker in instructions:
            if ling_pos < len(geez_indices) and ling_pos not in modified_geez_positions:
                actual_index = geez_indices[ling_pos]
                base_char = result[actual_index]
                
                order = self.language_utils.get_vowel_order_from_marker(marker)
                if order is not None:
                    result[actual_index] = self.language_utils.apply_vowel_to_consonant(base_char, order)
                    modified_geez_positions.add(ling_pos)
        
        # 🎯 STEP 4: Remove all CJK characters
        final_result = [c for c in result if not self._is_cjk_char(c)]
        reconstructed = ''.join(final_result)
        
        self.decode_cache[cache_key] = reconstructed
        return reconstructed
    
    def decode_line(self, line):
        """
        Decode a tokenized line - handles all boundary patterns:
        - ▁ markers (SentencePiece word boundaries)
        - :number, patterns (like ":4,")
        - Ethiopic punctuation (፡, ።, ፣, ፤, ፥, ፦, ፧, ፠, ፨)
        - Regular punctuation
        """
        if not line or not line.strip():
            return line
        
        # Split into tokens first
        tokens = line.strip().split()
        
        # 🎯 STEP 1: Identify word boundaries and group tokens
        word_groups = []
        current_group = []
        i = 0
        n = len(tokens)
        
        while i < n:
            token = tokens[i]
            
            # Check if this token is a boundary marker
            is_boundary = False
            
            # Case 1: Token ends with ▁
            if token.endswith('▁'):
                # Add clean token to current group
                current_group.append(token.replace('▁', ''))
                # Close the group
                if current_group:
                    word_groups.append(current_group)
                    current_group = []
                is_boundary = True
                i += 1
                continue
            
            # Case 2: Pattern like ":4," (number with colon and comma)
            if re.match(r'^:\d+,$', token):
                # Close previous group if exists
                if current_group:
                    word_groups.append(current_group)
                    current_group = []
                # Add this marker as its own group
                word_groups.append([token])
                i += 1
                continue
            
            # Case 3: Ethiopic punctuation that should be its own word
            if token in ['፡', '።', '፣', '፤', '፥', '፦', '፧', '፠', '፨']:
                if current_group:
                    word_groups.append(current_group)
                    current_group = []
                word_groups.append([token])
                i += 1
                continue
            
            # Regular token - add to current group
            current_group.append(token)
            i += 1
        
        # Add any remaining tokens as a final group
        if current_group:
            word_groups.append(current_group)
        
        # Debug: Show word groups if CJK present
        has_cjk = any(self._is_cjk_char(c) for group in word_groups for token in group for c in token)
        if has_cjk:
            print(f"\n{'='*60}")
            print(f">>> DECODER PROCESSING: '{line}'")
            print(f">>> Word groups: {word_groups}")
            print(f"{'='*60}")
        
        # 🎯 STEP 2: Process each word group
        result_words = []
        
        for group_idx, group in enumerate(word_groups):
            # If group has only one token and it's a special marker, keep as is
            if len(group) == 1:
                token = group[0]
                if re.match(r'^:\d+,$', token) or token in ['፡', '።', '፣', '፤', '፥', '፦', '፧', '፠', '፨']:
                    result_words.append(token)
                    continue
            
            # Join all tokens in the group to form the complete word
            combined_word = ''.join(group)
            
            # Split punctuation for better handling within the word
            combined_word = re.sub(r'([\[\](){},.;:!፡።፣፤፥፦፧፠፨])', r' \1 ', combined_word)
            word_tokens = combined_word.strip().split()
            
            # Process this word using stack approach
            word_stack = []
            j = 0
            m = len(word_tokens)
            
            while j < m:
                current = word_tokens[j]
                
                if any(self._is_cjk_char(c) for c in current):
                    # Check if this token has Ge'ez (attached case)
                    contains_geez = any(self.language_utils.is_letter_in_language(c) for c in current)
                    
                    if contains_geez:
                        # ATTACHED CASE: Ge'ez and CJK in same token
                        reconstructed = self._reconstruct_word(current)
                        word_stack.append(reconstructed)
                        j += 1
                    else:
                        # DISCONNECTED CASE: Pure CJK token(s)
                        # Collect all consecutive CJK tokens
                        cjk_tokens = [current]
                        j += 1
                        while j < m and any(self._is_cjk_char(c) for c in word_tokens[j]):
                            cjk_tokens.append(word_tokens[j])
                            j += 1
                        cjk_string = ''.join(cjk_tokens)
                        
                        # Look backwards in word_stack to find Ge'ez tokens that form the skeleton
                        skeleton_tokens = []
                        k = len(word_stack) - 1
                        while k >= 0:
                            stack_item = word_stack[k]
                            if any(self.language_utils.is_letter_in_language(c) for c in stack_item):
                                skeleton_tokens.insert(0, word_stack.pop(k))
                                k -= 1
                            else:
                                # Stop at first non-Ge'ez (punctuation)
                                break
                        
                        if skeleton_tokens:
                            # Combine skeleton tokens with CJK
                            combined = ''.join(skeleton_tokens) + cjk_string
                            reconstructed = self._reconstruct_word(combined)
                            word_stack.append(reconstructed)
                            if has_cjk:
                                print(f">>> Reconstructed '{combined}' → '{reconstructed}'")
                        else:
                            # No skeleton found - keep non-CJK parts
                            non_cjk = ''.join([c for c in cjk_string if not self._is_cjk_char(c)])
                            if non_cjk:
                                word_stack.append(non_cjk)
                else:
                    # Regular token (no CJK)
                    word_stack.append(current)
                    j += 1
            
            # Join the processed word
            result_words.append(''.join(word_stack))
        
        # Join all words with spaces
        final_output = ' '.join(result_words)
        
        # 🧼 Final cleanup: Remove structural markers and normalize
        final_output = final_output.replace('▁', '')
        final_output = re.sub(r'\s+([፡።፣፤፥፦፧፠፨\(\)\[\]\{\}.,;:!?])', r'\1', final_output)
        final_output = re.sub(r'([፡።፣፤፥፦፧፠፨\(\)\[\]\{\}.,;:!?])\s+', r'\1', final_output)
        final_output = re.sub(r'\(\s*:', r'(:', final_output)
        final_output = re.sub(r':\s*\)', r':)', final_output)
        final_output = re.sub(r'\s+', ' ', final_output)
        
        return final_output.strip()
    
    def decode_file(self, input_path, output_path):
        """Decode entire file."""
        self.logger.info(f"Decoding: {input_path}")
        line_count = 0
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
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
        
        self.logger.info(f"Complete: {line_count} lines")
        return line_count
    
    def compute_renyi_efficiency(self, tokenized_path: Path, alpha: float = 2.0, sample_lines: int = 10000):
        """Compute Rényi efficiency for tokenized corpus."""
        self.logger.info(f"Computing Rényi efficiency (α={alpha}) for {tokenized_path}")
        
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
        
        probs = np.array(list(token_counter.values())) / total_tokens
        
        if alpha == 1.0:
            renyi_entropy = -np.sum(probs * np.log(probs + 1e-10))
        else:
            renyi_entropy = (1 / (1 - alpha)) * np.log(np.sum(probs ** alpha) + 1e-10)
        
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
        """Evaluate round-trip accuracy."""
        self.logger.info(f"Evaluating round-trip accuracy")
        
        temp_path = tokenized_path.parent / f"temp_decoded_{tokenized_path.stem}.txt"
        self.decode_file(tokenized_path, temp_path)
        
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
        """Run full evaluation including Rényi efficiency and round-trip accuracy."""
        self.logger.info("=" * 60)
        self.logger.info("FULL EVALUATION")
        self.logger.info("=" * 60)
        
        results = {}
        
        for alpha in [0.5, 1.0, 2.0, 3.0]:
            results[f"renyi_alpha_{alpha}"] = self.compute_renyi_efficiency(
                tokenized_path, alpha=alpha
            )
        
        results["round_trip"] = self.evaluate_round_trip(original_path, tokenized_path)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Results saved to {output_path}")
        
        return results