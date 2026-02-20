"""
Base class for Ethiopic language utilities.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class EthiopicBase(ABC):
    """Base class for Ethiopic language utilities."""
    
    def __init__(self):
        self.name = "Ethiopic"
        self.code = "eth"
        
        # Ethiopic syllabary ranges
        self.ETHIOPIC_RANGES = [
            (0x1200, 0x1248),  # Ge'ez
            (0x124A, 0x124D),
            (0x1250, 0x1256),
            (0x1258, 0x1258),
            (0x125A, 0x125D),
            (0x1260, 0x1288),
            (0x128A, 0x128D),
            (0x1290, 0x12B0),
            (0x12B2, 0x12B5),
            (0x12B8, 0x12BE),
            (0x12C0, 0x12C0),
            (0x12C2, 0x12C5),
            (0x12C8, 0x12D6),
            (0x12D8, 0x1310),
            (0x1312, 0x1315),
            (0x1318, 0x135A),
            (0x1360, 0x137C),  # Punctuation
            (0x1380, 0x1399),  # Supplements
            (0x2D80, 0x2D96),  # Extended
            (0x2DA0, 0x2DA6),
            (0x2DA8, 0x2DAE),
            (0x2DB0, 0x2DB6),
            (0x2DB8, 0x2DBE),
            (0x2DC0, 0x2DC6),
            (0x2DC8, 0x2DCE),
            (0x2DD0, 0x2DD6),
            (0x2DD8, 0x2DDE),
            (0xAB01, 0xAB06),  # Extended-A
            (0xAB09, 0xAB0E),
            (0xAB11, 0xAB16),
            (0xAB20, 0xAB26),
            (0xAB28, 0xAB2E),
        ]
        
        # Vowel markers in Ethiopic
        self.VOWELS = {
            'ə': 0x00,  # 1st order (default)
            'u': 0x01,  # 2nd order
            'i': 0x02,  # 3rd order
            'a': 0x03,  # 4th order
            'e': 0x04,  # 5th order
            'ɨ': 0x05,  # 6th order
            'o': 0x06,  # 7th order
        }
        
        # Vowel to suffix mapping
        self.VOWEL_TO_SUFFIX = {
            'ə': '',    # Base form
            'u': 'ቱ',   # Often appears as suffix
            'i': 'ቲ',
            'a': 'ታ',
            'e': 'ቴ',
            'ɨ': 'ት',
            'o': 'ቶ',
        }
    
    def is_ethiopic(self, char: str) -> bool:
        """Check if character is in Ethiopic range."""
        if len(char) != 1:
            return False
        
        code = ord(char)
        for start, end in self.ETHIOPIC_RANGES:
            if start <= code <= end:
                return True
        return False
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Ethiopic text.
        Override in language-specific classes.
        """
        # Remove non-Ethiopic characters by default
        return ''.join(c for c in text if self.is_ethiopic(c) or c.isspace())
    
    @abstractmethod
    def syllable_breaking(self, word: str) -> str:
        """
        Break word into syllables.
        Returns: Space-separated syllables
        """
        pass
    
    @abstractmethod
    def extract_consonant_skeleton(self, broken_word: str) -> Tuple[str, List[Tuple[int, str]]]:
        """
        Extract consonant skeleton and vowel markers.
        
        Args:
            broken_word: Space-separated syllables
            
        Returns:
            Tuple of (consonant_skeleton, list of (consonant_index, vowel))
        """
        pass
    
    def save_additional_corpora_for_evaluation(self, text_processor):
        """
        Save additional corpora for evaluation.
        Override in language-specific classes if needed.
        """
        pass