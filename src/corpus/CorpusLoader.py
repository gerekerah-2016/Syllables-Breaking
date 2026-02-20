"""
Load corpus from various sources.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import os
from pathlib import Path
from typing import List, Optional

try:
    from datasets import load_dataset
    HAS_HF = True
except ImportError:
    HAS_HF = False


class CorpusLoader:
    """Load corpus from local directory or Hugging Face."""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.file_count = 0
    
    def load_from_directory(self, directory_path: str, extension: str = ".txt") -> List[str]:
        """Load corpus from local directory."""
        if self.logger:
            self.logger.info(f"Loading from directory: {directory_path}")
        
        all_words = []
        path = Path(directory_path)
        
        if not path.exists():
            if self.logger:
                self.logger.warning(f"Directory not found: {directory_path}")
            return []
        
        for file_path in path.rglob(f"*{extension}"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    words = text.split()
                    all_words.extend(words)
                    self.file_count += 1
                    
                    if self.logger and self.file_count % 10 == 0:
                        self.logger.info(f"  Loaded {self.file_count} files, {len(all_words)} words")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"  Error reading {file_path}: {e}")
        
        if self.logger:
            self.logger.info(f"✓ Loaded {self.file_count} files, {len(all_words)} words")
        
        return all_words
    
    def load_from_huggingface(self, dataset_name: str, split: str = "train", text_field: str = "text") -> List[str]:
        """Load corpus from Hugging Face dataset."""
        if not HAS_HF:
            raise ImportError("Hugging Face datasets not installed")
        
        if self.logger:
            self.logger.info(f"Loading from Hugging Face: {dataset_name}")
        
        dataset = load_dataset(dataset_name, split=split)
        
        words = []
        for item in dataset:
            if text_field in item:
                text = item[text_field]
            else:
                for key, value in item.items():
                    if isinstance(value, str):
                        text = value
                        break
                else:
                    continue
            
            words.extend(text.split())
        
        if self.logger:
            self.logger.info(f"✓ Loaded {len(words)} words from Hugging Face")
        
        return words
    
    def load(self, source: str, is_huggingface: bool = False, **kwargs) -> List[str]:
        """Load corpus from source."""
        if is_huggingface:
            return self.load_from_huggingface(
                source,
                split=kwargs.get('split', 'train'),
                text_field=kwargs.get('text_field', 'text')
            )
        else:
            return self.load_from_directory(
                source,
                extension=kwargs.get('extension', '.txt')
            )