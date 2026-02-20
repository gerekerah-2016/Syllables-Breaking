"""
Extract words from corpus and create frequency dictionaries.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import json
import os
import re
from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface
from src.logger import get_logger
from src.utils.path_utils import get_words_dict_dir


class CorpusWordsExtractor:
    """Extract words from corpus and create frequency dictionaries."""
    
    def __init__(self, language_utils: LanguageUtilsInterface):
        self.language_utils = language_utils

    def convert_corpus_to_words_dict_file(self, corpus, output_filename):
        """
        Convert corpus to word frequency dictionary and save to file.
        
        Args:
            corpus: Corpus data (can be dict with 'text' key or list)
            output_filename: Name for output file
        """
        get_logger().info(f"Converting corpus to word dictionary: {output_filename}")
        
        if isinstance(corpus, dict) and 'text' in corpus:
            words = self.get_words_from_corpus(corpus["text"])
        elif isinstance(corpus, list):
            words = self.get_words_from_corpus(corpus)
        else:
            get_logger().warning(f"Unexpected corpus type: {type(corpus)}")
            return
            
        os.makedirs(get_words_dict_dir(), exist_ok=True)
        output_path = f'{get_words_dict_dir()}/{output_filename}.json'
        
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(words, file, indent='\t', ensure_ascii=False)
        
        get_logger().info(f"Saved word dictionary with {len(words)} unique words to {output_path}")

    def get_words_from_corpus(self, articles_text):
        """
        Extract words from corpus text with frequencies.
        
        Args:
            articles_text: List of article texts
            
        Returns:
            Dictionary mapping words to frequencies
        """
        words = {}
        total_articles = len(articles_text)
        
        for index, article_text in enumerate(articles_text):
            if not article_text or not isinstance(article_text, str):
                continue
                
            # Remove diacritics if language supports it
            if hasattr(self.language_utils, 'remove_diacritics'):
                article_text = self.language_utils.remove_diacritics(article_text)
            
            # Split into words using common delimiters
            article_words = re.split(r'[.\s\n\-,\:\"\(\)]', article_text)
            
            for word in article_words:
                word = word.strip()
                if not word:
                    continue
                    
                # Replace final letters if language supports it
                if hasattr(self.language_utils, 'replace_final_letters'):
                    word = self.language_utils.replace_final_letters(word)
                    
                words[word] = words.get(word, 0) + 1

            if (index + 1) % 10000 == 0:
                get_logger().info(f'Processed {index + 1}/{total_articles} articles, found {len(words)} unique words so far')
        
        get_logger().info(f'Finished processing {total_articles} articles, found {len(words)} unique words')
        return words