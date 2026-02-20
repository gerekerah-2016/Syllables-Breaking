"""
Splinter trainer for learning reduction patterns from corpus.
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
    Learns reduction patterns from Ethiopic corpus.
    Creates mappings like '0:[a]' → '⟨1⟩' for vowel reductions.
    """
    
    def __init__(self, language_utils: LanguageUtilsInterface):
        self.language_utils = language_utils
        self.encode_map = {}  # Maps reduction keys to tags (e.g., "0:[a]" → "⟨1⟩")
        self.decode_map = {}  # Maps tags to reduction keys (e.g., "⟨1⟩" → "0:[a]")
        self.counter = 1
        self.stats = {
            'total_words': 0,
            'total_syllables': 0,
            'vowel_counts': {},
            'word_lengths': [],
            'reduction_frequencies': {}
        }
        # Initialize vowel counts
        if hasattr(language_utils, 'VOWEL_SYMBOLS'):
            for v in language_utils.VOWEL_SYMBOLS.values():
                self.stats['vowel_counts'][v] = 0

    def train(self, dataset_path: str, dataset_name: str, letters_for_reductions: [str] = None):
        """
        Train splinter reduction rules on a corpus.
        
        Args:
            dataset_path: Path to dataset (local directory)
            dataset_name: Name of dataset
            letters_for_reductions: Optional subset of letters to consider
            
        Returns:
            Tuple of (reductions_map, new_unicode_chars_map, new_unicode_chars_inverted_map)
        """
        get_logger().info(f"Starting splinter training on {dataset_path}/{dataset_name}")
        
        # Get word frequency dictionary
        words_dict = self.get_word_dict(dataset_path, dataset_name)
        
        if not words_dict:
            get_logger().error("No words found in corpus!")
            return {}, {}, {}
        
        get_logger().info(f"Found {len(words_dict)} unique words in corpus")
        
        # Process each word to learn reductions
        processed = 0
        for word, freq in words_dict.items():
            if len(word) < 2 or freq < 10:  # Filter rare and short words
                continue
            
            try:
                self.process_word(word)
                processed += 1
                if processed % 10000 == 0:
                    get_logger().info(f"  Processed {processed} words, found {len(self.encode_map)} rules")
            except Exception as e:
                get_logger().warning(f"Error processing '{word}': {e}")
        
        get_logger().info(f"✓ Learned {len(self.encode_map)} reduction rules")
        get_logger().info(f"  Token range: ⟨1⟩ to ⟨{self.counter-1}⟩")
        
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

    def process_word(self, word):
        """
        Process a single word to learn reduction patterns.
        
        This method:
        1. Breaks the word into syllables (consonant + vowel marker)
        2. Extracts the consonant skeleton
        3. Creates reduction keys for each vowel (e.g., "0:[a]")
        4. Assigns tags (e.g., "⟨1⟩") to each unique reduction
        
        Args:
            word: Original Ge'ez word
        """
        # Step 1: Syllable breaking
        broken = self.language_utils.syllable_break(word)
        
        # Step 2: Extract consonant skeleton and vowel data
        consonants = []
        vowel_data = []
        
        i = 0
        length = len(broken)
        while i < length:
            char = broken[i]
            
            # Check if this is a consonant (Ethiopic letter)
            if self.language_utils.is_letter_in_language(char):
                consonants.append(char)
                curr_cons_idx = len(consonants) - 1
                
                # Look ahead to see if there's a vowel marker
                # Vowel markers start with '[' and end with ']'
                if i + 1 < length and broken[i+1] == '[':
                    # Find the closing bracket
                    end_idx = broken.find(']', i+1)
                    if end_idx != -1:
                        # Extract the FULL vowel marker including brackets
                        # This gives "[a]" not just "["
                        vowel_marker = broken[i+1:end_idx+1]
                        
                        vowel_data.append((curr_cons_idx, vowel_marker))
                        
                        self.stats['total_syllables'] += 1
                        if vowel_marker in self.stats['vowel_counts']:
                            self.stats['vowel_counts'][vowel_marker] += 1
                        
                        # Skip ahead past the vowel marker
                        i = end_idx + 1
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        
        # Step 3: Create reductions for each vowel
        for cons_idx, v_marker in vowel_data:
            key = f"{cons_idx}:{v_marker}"
            
            # Track frequency
            self.stats['reduction_frequencies'][key] = self.stats['reduction_frequencies'].get(key, 0) + 1
            
            # Assign new tag if this reduction hasn't been seen before
            if key not in self.encode_map:
                tag = f"⟨{self.counter}⟩"
                self.encode_map[key] = tag
                self.decode_map[tag] = key
                self.counter += 1
        
        self.stats['total_words'] += 1
        self.stats['word_lengths'].append(len(consonants))

    def get_word_dict(self, dataset_path, dataset_name):
        """
        Get or create word frequency dictionary from the corpus.
        
        Args:
            dataset_path: Path to dataset (local directory)
            dataset_name: Name of dataset
            
        Returns:
            Dictionary mapping words to frequencies
        """
        corpus_name = get_corpus_name(dataset_path, dataset_name)
        dict_path = os.path.join(get_words_dict_dir(), f'{corpus_name}.json')
        
        if not os.path.exists(dict_path):
            get_logger().info(f'Creating word dictionary from {dataset_path}')
            
            # Load from local files
            texts = self.load_local_files(dataset_path)
            words_dict = {}
            
            for text in texts:
                # Split into words and count frequencies
                for word in text.split():
                    word = word.strip()
                    if word:
                        words_dict[word] = words_dict.get(word, 0) + 1
            
            # Save dictionary
            os.makedirs(get_words_dict_dir(), exist_ok=True)
            with open(dict_path, 'w', encoding='utf-8') as f:
                json.dump(words_dict, f, indent='\t', ensure_ascii=False)
            get_logger().info(f"Saved word dictionary with {len(words_dict)} entries")
        
        # Load from file
        with open(dict_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_local_files(self, directory_path):
        """
        Load text from local files recursively.
        
        Args:
            directory_path: Path to directory containing .txt files
            
        Returns:
            List of text contents
        """
        texts = []
        if not os.path.exists(directory_path):
            get_logger().warning(f"Directory not found: {directory_path}")
            return texts
        
        file_count = 0
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.txt'):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content.strip():
                                texts.append(content)
                                file_count += 1
                                if file_count % 10 == 0:
                                    get_logger().info(f"  Loaded {file_count} files...")
                    except Exception as e:
                        get_logger().warning(f"Error reading {path}: {e}")
        
        get_logger().info(f"Loaded {file_count} files from {directory_path}")
        return texts

    def create_reductions_map(self):
        """
        Create reductions map in the format expected by the pipeline.
        
        Returns:
            Dictionary mapping word lengths to reduction probabilities
        """
        reductions_map = {}
        
        # Group reductions by word length
        for key in self.encode_map.keys():
            # Parse key like "0:[a]"
            if ':' in key:
                idx_str, marker = key.split(':', 1)
                try:
                    idx = int(idx_str)
                    # Estimate word length from index (rough approximation)
                    word_length = idx + 3  # Minimum word length is 3 consonants
                    
                    if word_length not in reductions_map:
                        reductions_map[word_length] = {}
                    
                    # Add reduction with equal probability (can be refined later)
                    reductions_map[word_length][key] = 1.0
                except ValueError:
                    continue
        
        # Add base consonants for length 1
        if hasattr(self.language_utils, 'BASE_CONSONANTS'):
            reductions_map[1] = {cons: 1.0 for cons in self.language_utils.BASE_CONSONANTS}
        
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
        
        # Save to logs directory - CREATE IT FIRST
        logs_dir = get_logs_dir()
        os.makedirs(logs_dir, exist_ok=True)
        logs_path = os.path.join(logs_dir, f'{file_name}.json')
        with open(logs_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent='\t', ensure_ascii=False)
        
        get_logger().info(f"Saved {file_name} to {path}")

    def print_statistics(self):
        """Print training statistics."""
        print("\n" + "=" * 60)
        print("SPLINTER TRAINING STATISTICS")
        print("=" * 60)
        print(f"Total words processed: {self.stats['total_words']:,}")
        print(f"Total syllables: {self.stats['total_syllables']:,}")
        print(f"Unique reduction rules: {len(self.encode_map)}")
        print(f"Tag range: ⟨1⟩ to ⟨{self.counter-1}⟩")
        
        print("\nVowel distribution:")
        # Sort by frequency
        sorted_vowels = sorted(self.stats['vowel_counts'].items(), 
                              key=lambda x: x[1], reverse=True)
        for vowel, count in sorted_vowels:
            if count > 0 and self.stats['total_syllables'] > 0:
                percentage = (count / self.stats['total_syllables']) * 100
                print(f"  {vowel}: {count:,} ({percentage:.1f}%)")
        
        print("\nTop 10 most frequent reductions:")
        sorted_red = sorted(self.stats['reduction_frequencies'].items(), 
                           key=lambda x: x[1], reverse=True)[:10]
        for red, freq in sorted_red:
            tag = self.encode_map.get(red, "unknown")
            print(f"  {red} → {tag}: {freq} times")
        
        print("=" * 60)