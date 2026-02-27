"""
Splinter trainer for learning reduction patterns from Ethiopic/Ge'ez corpus.
Specifically designed for Ge'ez/Amharic abugida writing system.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from datasets import load_dataset

from src.CorpusWordsExtractor import CorpusWordsExtractor
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface
from src.logger import get_logger
from src.utils.path_utils import get_logs_dir, get_raw_data_dir, get_splinter_dir, get_words_dict_dir
from src.utils.utils import get_words_dict_by_length, get_permutation, get_corpus_name


class SplinterTrainer:
    """
    Learns reduction patterns from Ethiopic/Ge'ez corpus.
    Specifically designed for abugida writing system where:
    - Each character is consonant + vowel
    - Base form is 1st order [ə]
    - Vowel patterns appear at specific positions
    
    Creates mappings like '0:[a]' → CJK character directly (e.g., '一').
    Only creates tags for patterns that appear frequently enough.
    """
    
    def __init__(self, language_utils: LanguageUtilsInterface, min_frequency: int = 5):
        """
        Initialize SplinterTrainer for Ge'ez/Amharic.
        
        Args:
            language_utils: Language utilities instance
            min_frequency: Minimum frequency for a reduction to get a tag
        """
        self.language_utils = language_utils
        self.encode_map = {}  # Maps reduction keys to CJK chars (e.g., "0:[a]" → "一")
        self.decode_map = {}  # Maps CJK chars to reduction keys (e.g., "一" → "0:[a]")
        self.counter = 1
        self.min_frequency = min_frequency
        self.stats = {
            'total_words': 0,
            'total_syllables': 0,
            'vowel_counts': {},
            'word_lengths': [],
            'reduction_frequencies': {},
            'vowel_order_stats': {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0},
            'position_vowel_counts': defaultdict(lambda: defaultdict(int))
        }
        # Initialize vowel counts - include ALL vowel symbols including [ə]
        if hasattr(language_utils, 'VOWEL_SYMBOLS'):
            for v in language_utils.VOWEL_SYMBOLS.values():
                self.stats['vowel_counts'][v] = 0

    def train(self, dataset_path: str, dataset_name: str, letters_for_reductions: [str] = None):
        """
        Train splinter reduction rules on a Ge'ez/Amharic corpus.
        
        Ge'ez-specific algorithm:
        1. Group words by consonant skeleton length
        2. For each position in skeleton, find common vowel patterns
        3. Create single-character CJK tags for frequent vowel patterns
        
        Args:
            dataset_path: Path to dataset (local directory)
            dataset_name: Name of dataset
            letters_for_reductions: Optional subset of letters to consider
            
        Returns:
            Tuple of (reductions_map, new_unicode_chars_map, new_unicode_chars_inverted_map)
        """
        get_logger().info(f"Starting Ge'ez-specific splinter training on {dataset_path}/{dataset_name}")
        
        # Get word frequency dictionary
        words_dict = self.get_word_dict(dataset_path, dataset_name)
        
        if not words_dict:
            get_logger().error("No words found in corpus!")
            return {}, {}, {}
        
        get_logger().info(f"Found {len(words_dict)} unique words in corpus")
        
        # ============================================================
        # GE'EZ-SPECIFIC: Group words by consonant skeleton length
        # ============================================================
        get_logger().info("Grouping words by consonant skeleton length...")
        
        # Group words by their consonant skeleton
        words_by_skeleton = defaultdict(list)  # skeleton -> [(word, freq, vowel_orders)]
        words_by_skeleton_length = defaultdict(list)  # length -> [skeletons]
        
        for word, freq in words_dict.items():
            if len(word) < 2 or freq < 10:  # Filter rare and short words
                continue
            
            # Extract base consonants and vowel orders
            base_consonants = []
            vowel_orders = []
            
            for char in word:
                if self.language_utils.is_letter_in_language(char):
                    base, order = self.language_utils.get_base_and_order(char)
                    base_consonants.append(base)
                    vowel_orders.append(order)
                    # Update vowel order statistics
                    if order in self.stats['vowel_order_stats']:
                        self.stats['vowel_order_stats'][order] += freq
                else:
                    # Skip non-Ge'ez characters
                    continue
            
            if not base_consonants:
                continue
                
            skeleton = ''.join(base_consonants)
            words_by_skeleton[skeleton].append((word, freq, vowel_orders))
            words_by_skeleton_length[len(skeleton)].append(skeleton)
            
            # Update total words count
            self.stats['total_words'] += 1
            self.stats['word_lengths'].append(len(base_consonants))
        
        get_logger().info(f"  Found {len(words_by_skeleton)} unique consonant skeletons")
        
        # ============================================================
        # FIRST PASS: Count vowel patterns at each position
        # ============================================================
        get_logger().info("First pass: Counting vowel patterns at each position...")
        
        # For each skeleton length, track which vowels appear at each position
        position_vowel_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # position_vowel_counts[length][position][vowel_order] = count
        
        for skeleton, word_list in words_by_skeleton.items():
            length = len(skeleton)
            
            for word, freq, vowel_orders in word_list:
                for pos, order in enumerate(vowel_orders):
                    # Only count non-base vowels (order != 0) for reductions
                    # Base form [ə] is the skeleton itself, not a reduction
                    if order != 0:
                        position_vowel_counts[length][pos][order] += freq
                        
                        # Also track overall vowel counts
                        vowel_marker = self.language_utils.VOWEL_SYMBOLS.get(order, f"[{order}]")
                        self.stats['vowel_counts'][vowel_marker] = \
                            self.stats['vowel_counts'].get(vowel_marker, 0) + freq
        
        # ============================================================
        # SECOND PASS: Identify frequent vowel patterns
        # ============================================================
        get_logger().info("Second pass: Identifying frequent vowel patterns...")
        
        # For each skeleton length and position, find most common vowel patterns
        pattern_candidates = []  # (length, position, order, count)
        
        for length in position_vowel_counts:
            for pos in position_vowel_counts[length]:
                total_at_position = sum(position_vowel_counts[length][pos].values())
                
                for order, count in position_vowel_counts[length][pos].items():
                    # Calculate frequency percentage
                    percentage = (count / total_at_position) * 100 if total_at_position > 0 else 0
                    
                    # Store candidate if it meets minimum frequency
                    if count >= self.min_frequency:
                        pattern_candidates.append((length, pos, order, count, percentage))
                        
                        # Track reduction frequency for statistics
                        vowel_marker = self.language_utils.VOWEL_SYMBOLS.get(order, f"[{order}]")
                        reduction_key = f"{pos}:{vowel_marker}"
                        self.stats['reduction_frequencies'][reduction_key] = \
                            self.stats['reduction_frequencies'].get(reduction_key, 0) + count
        
        # Sort candidates by count (highest first)
        pattern_candidates.sort(key=lambda x: x[3], reverse=True)
        
        get_logger().info(f"  Found {len(pattern_candidates)} frequent vowel patterns")
        
        # ============================================================
        # DIRECT CJK ASSIGNMENT - NO BRACKETS, NO TAG MAPPER
        # ============================================================
        get_logger().info(f"Creating direct CJK mappings for patterns with frequency >= {self.min_frequency}...")
        
        # Get all required reduction keys from pattern candidates
        all_required_reduction_keys = []
        for length, pos, order, count, percentage in pattern_candidates:
            vowel_marker = self.language_utils.VOWEL_SYMBOLS.get(order, f"[{order}]")
            reduction_key = f"{pos}:{vowel_marker}"
            all_required_reduction_keys.append(reduction_key)
        
        # Remove duplicates while preserving order
        seen = set()
        all_required_reduction_keys = [x for x in all_required_reduction_keys if not (x in seen or seen.add(x))]
        
        get_logger().info(f"Creating {len(all_required_reduction_keys)} direct CJK mappings")
        
        # DIRECT CJK ASSIGNMENT - start at U+4E00
        self.encode_map = {}
        self.decode_map = {}
        
        for i, key in enumerate(all_required_reduction_keys):
            # Start at U+4E00 (First CJK character) and increment
            cjk_char = chr(0x4E00 + i)
            
            # Store in maps
            self.encode_map[key] = cjk_char
            self.decode_map[cjk_char] = key
        
        get_logger().info(f"  Created {len(self.encode_map)} direct CJK mappings")
        get_logger().info(f"  CJK range: U+4E00 to U+{0x4E00 + len(self.encode_map) - 1:04X}")
        
        # ============================================================
        # Also learn patterns for suffixes/prefixes (multi-character)
        # ============================================================
        get_logger().info("Learning common affix patterns...")
        
        # Group words by their skeleton to find common affixes
        skeleton_groups = defaultdict(list)
        for skeleton, word_list in words_by_skeleton.items():
            for word, freq, vowel_orders in word_list:
                skeleton_groups[skeleton].append((word, freq, vowel_orders))
        
        # For skeletons that appear with multiple vowel patterns, learn the variations
        affix_patterns = defaultdict(int)
        
        for skeleton, word_list in skeleton_groups.items():
            if len(word_list) >= 3:  # At least 3 variations of this root
                # This skeleton has multiple forms - likely a productive root
                for word, freq, vowel_orders in word_list:
                    # Create a pattern key that captures the vowel differences
                    pattern_key = []
                    for pos, order in enumerate(vowel_orders):
                        if order != 0:
                            vowel_marker = self.language_utils.VOWEL_SYMBOLS.get(order, f"[{order}]")
                            pattern_key.append(f"{pos}:{vowel_marker}")
                    
                    if pattern_key:
                        pattern_str = ','.join(pattern_key)
                        affix_patterns[pattern_str] += freq
        
        # Create direct CJK mappings for common affix patterns
        affix_tags_created = 0
        for pattern, count in sorted(affix_patterns.items(), key=lambda x: x[1], reverse=True)[:20]:
            if count >= self.min_frequency * 2 and pattern not in self.encode_map:  # Higher threshold for multi-patterns
                cjk_char = chr(0x4E00 + len(self.encode_map))
                self.encode_map[pattern] = cjk_char
                self.decode_map[cjk_char] = pattern
                affix_tags_created += 1
                get_logger().debug(f"  Created CJK char '{cjk_char}' for affix pattern '{pattern}' (count: {count})")
        
        if affix_tags_created > 0:
            get_logger().info(f"  Created {affix_tags_created} affix pattern mappings")
        
        get_logger().info(f"✓ Learned {len(self.encode_map)} total reduction rules")
        
        # Print statistics
        self.print_statistics()
        
        # Create the three maps expected by the pipeline
        reductions_map = self.create_reductions_map()
        new_unicode_chars_map = self.encode_map.copy()
        new_unicode_chars_inverted_map = self.decode_map.copy()
        
        # Save results
        self.save_result_file("reductions_map", reductions_map)
        self.save_result_file("new_unicode_chars", new_unicode_chars_map)
        self.save_result_file("new_unicode_chars_inverted", new_unicode_chars_inverted_map)
        
        return reductions_map, new_unicode_chars_map, new_unicode_chars_inverted_map

    def get_word_dict(self, dataset_path, dataset_name):
        """
        Get or create word frequency dictionary from the corpus.
        FIXED: Process files line by line to avoid MemoryError
        """
        corpus_name = get_corpus_name(dataset_path, dataset_name)
        dict_path = os.path.join(get_words_dict_dir(), f'{corpus_name}.json')
        
        if not os.path.exists(dict_path):
            get_logger().info(f'Creating word dictionary from {dataset_path}')
            
            if not os.path.exists(dataset_path):
                get_logger().error(f"Dataset path not found: {dataset_path}")
                return {}
            
            words_dict = {}
            file_count = 0
            total_words = 0
            
            # Walk through all files
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        try:
                            # Process file line by line - THIS PREVENTS MEMORYERROR
                            with open(file_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    
                                    # Split line into words and count
                                    for word in line.split():
                                        word = word.strip()
                                        if word:
                                            words_dict[word] = words_dict.get(word, 0) + 1
                                            total_words += 1
                            
                            file_count += 1
                            if file_count % 10 == 0:
                                get_logger().info(f"  Processed {file_count} files, {len(words_dict)} unique words, {total_words} total words")
                                
                        except Exception as e:
                            get_logger().warning(f"Error reading {file_path}: {e}")
            
            # Save dictionary
            os.makedirs(get_words_dict_dir(), exist_ok=True)
            with open(dict_path, 'w', encoding='utf-8') as f:
                json.dump(words_dict, f, indent='\t', ensure_ascii=False)
            
            get_logger().info(f"✅ Saved word dictionary with {len(words_dict)} unique words from {file_count} files")
        
        # Load from file
        with open(dict_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_local_files(self, directory_path):
        """
        Kept for compatibility - but not used in new version
        """
        get_logger().warning("load_local_files is deprecated - use get_word_dict instead")
        return []

    def create_reductions_map(self):
        """
        Create reductions map in the format expected by the pipeline.
        
        Returns:
            Dictionary mapping word lengths to reduction probabilities
        """
        reductions_map = {}
        
        # Group reductions by word length
        for key in self.encode_map.keys():
            # Parse key like "0:[a]" or multi-pattern "0:[a],2:[u]"
            if ':' in key:
                if ',' in key:
                    # Multi-pattern (affix)
                    # Estimate length from first pattern
                    first_pattern = key.split(',')[0]
                    if ':' in first_pattern:
                        idx_str = first_pattern.split(':')[0]
                        try:
                            idx = int(idx_str)
                            word_length = idx + 3
                            if word_length not in reductions_map:
                                reductions_map[word_length] = {}
                            reductions_map[word_length][key] = 1.0
                        except ValueError:
                            continue
                else:
                    # Single pattern
                    idx_str, marker = key.split(':', 1)
                    try:
                        idx = int(idx_str)
                        word_length = idx + 3
                        
                        if word_length not in reductions_map:
                            reductions_map[word_length] = {}
                        
                        # Use frequency to calculate probability
                        freq = self.stats['reduction_frequencies'].get(key, 1)
                        reductions_map[word_length][key] = float(freq)
                    except ValueError:
                        continue
        
        # Normalize probabilities by word length
        for length in reductions_map:
            total = sum(reductions_map[length].values())
            if total > 0:
                for key in reductions_map[length]:
                    reductions_map[length][key] /= total
        
        # Add base consonants for length 1
        if hasattr(self.language_utils, 'BASE_CONSONANTS'):
            reductions_map[1] = {cons: 1.0/len(self.language_utils.BASE_CONSONANTS) 
                                for cons in self.language_utils.BASE_CONSONANTS}
        
        return reductions_map

    def save_result_file(self, file_name, data):
        """
        Save result data to file in both splinter dir and logs dir.
        
        Args:
            file_name: Name of file (without extension)
            data: Data to save
        """
        # Save to splinter directory
        splinter_dir = get_splinter_dir()
        os.makedirs(splinter_dir, exist_ok=True)
        path = os.path.join(splinter_dir, f'{file_name}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent='\t', ensure_ascii=False)
        
        # Save to logs directory
        logs_dir = get_logs_dir()
        os.makedirs(logs_dir, exist_ok=True)
        logs_path = os.path.join(logs_dir, f'{file_name}.json')
        with open(logs_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent='\t', ensure_ascii=False)
        
        get_logger().info(f"Saved {file_name} to {path}")

    def print_statistics(self):
        """Print training statistics."""
        print("\n" + "=" * 70)
        print("GE'EZ SPLINTER TRAINING STATISTICS")
        print("=" * 70)
        print(f"Total words processed: {self.stats['total_words']:,}")
        print(f"Unique reduction rules: {len(self.encode_map)}")
        print(f"Direct CJK mapping range: U+4E00 to U+{0x4E00 + len(self.encode_map) - 1:04X}")
        print(f"Min frequency threshold: {self.min_frequency}")
        
        print("\nVowel Order Distribution:")
        total_vowels = sum(self.stats['vowel_order_stats'].values())
        if total_vowels > 0:
            for order in range(12):
                count = self.stats['vowel_order_stats'].get(order, 0)
                if count > 0:
                    percentage = (count / total_vowels) * 100
                    vowel_name = self.language_utils.VOWEL_SYMBOLS.get(order, f"[{order}]")
                    print(f"  Order {order} {vowel_name}: {count:,} ({percentage:.1f}%)")
        
        print("\nVowel marker distribution:")
        sorted_vowels = sorted(self.stats['vowel_counts'].items(), 
                              key=lambda x: x[1], reverse=True)
        for vowel, count in sorted_vowels[:10]:
            if count > 0 and self.stats['total_syllables'] > 0:
                percentage = (count / self.stats['total_syllables']) * 100
                print(f"  {vowel}: {count:,} ({percentage:.1f}%)")
        
        print("\nTop 15 most frequent reduction patterns (mapped to CJK chars):")
        sorted_red = sorted(self.stats['reduction_frequencies'].items(), 
                           key=lambda x: x[1], reverse=True)[:15]
        for red, freq in sorted_red:
            cjk_char = self.encode_map.get(red, "?")
            print(f"  {red:15} → '{cjk_char}' (U+{ord(cjk_char):04X}) : {freq:6} times")
        
        print("\n" + "=" * 70)