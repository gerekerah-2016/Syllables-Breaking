"""
Ge'ez language utilities.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import re
from typing import List, Tuple
from src.language_utils.EthiopicBase import EthiopicBase


class GeezUtils(EthiopicBase):
    """Ge'ez language utilities."""
    
    def __init__(self):
        super().__init__()
        self.name = "Ge'ez"
        self.code = "ge"
        
        # Ge'ez syllabary mapping (consonant + vowel)
        self.CONSONANTS = {
            'ሀ': 'h', 'ለ': 'l', 'ሐ': 'ħ', 'መ': 'm', 'ሠ': 'ś',
            'ረ': 'r', 'ሰ': 's', 'ሸ': 'š', 'ቀ': 'q', 'በ': 'b',
            'ተ': 't', 'ቸ': 'č', 'ኀ': 'ḫ', 'ነ': 'n', 'ኘ': 'ñ',
            'አ': 'ʾ', 'ከ': 'k', 'ኸ': 'x', 'ወ': 'w', 'ዐ': 'ʿ',
            'ዘ': 'z', 'ዠ': 'ž', 'የ': 'y', 'ደ': 'd', 'ዸ': 'ḍ',
            'ጀ': 'ǧ', 'ገ': 'g', 'ጠ': 'ṭ', 'ጨ': 'č̣', 'ጰ': 'p̣',
            'ጸ': 'ṣ', 'ፀ': 'ṣ́', 'ፈ': 'f', 'ፐ': 'p'
        }
        
        # Syllable patterns
        self.SYLLABLE_PATTERN = re.compile(r'([ሀ-ፐ])([ኧኡኢኣኤእኦ]|ኧ)?')
    
    def normalize_text(self, text: str) -> str:
        """Normalize Ge'ez text."""
        # First get base normalization
        normalized = super().normalize_text(text)
        
        # Ge'ez-specific normalizations
        replacements = {
            '፡': ' ',  # Word separator to space
            '።': '.',  # Period
            '፣': ',',  # Comma
            '፤': ';',  # Semicolon
            '፥': ':',  # Colon
            '፦': '«',  # Left quotation
            '፧': '»',  # Right quotation
            '፨': ' ',  # Section separator
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def syllable_breaking(self, word: str) -> str:
        """
        Break Ge'ez word into syllables.
        Ge'ez syllables are already represented as single characters.
        """
        if not word:
            return ""
        
        # Each character is a syllable in Ge'ez
        syllables = []
        for char in word:
            if self.is_ethiopic(char):
                syllables.append(char)
            else:
                # Keep spaces but don't break on them
                if char.isspace():
                    syllables.append(char)
        
        return ' '.join(syllables)
    
    def extract_consonant_skeleton(self, broken_word: str) -> Tuple[str, List[Tuple[int, str]]]:
        """
        Extract consonant skeleton from Ge'ez syllables.
        Each syllable is already a consonant+vowel combination.
        """
        syllables = broken_word.split()
        consonant_skeleton = []
        vowel_data = []
        
        for i, syllable in enumerate(syllables):
            if not self.is_ethiopic(syllable):
                consonant_skeleton.append(syllable)
                continue
            
            # In Ge'ez, the consonant and vowel are in one character
            # We need to determine the vowel from the character
            consonant = self._get_consonant(syllable)
            vowel = self._get_vowel(syllable)
            
            consonant_skeleton.append(consonant)
            vowel_data.append((i, vowel))
        
        return ''.join(consonant_skeleton), vowel_data
    
    def _get_consonant(self, syllable: str) -> str:
        """Extract consonant from Ge'ez syllable."""
        # This is simplified - real mapping would be more complex
        base_consonants = {
            'ሀ': 'ሀ', 'ለ': 'ለ', 'ሐ': 'ሐ', 'መ': 'መ', 'ሠ': 'ሠ',
            'ረ': 'ረ', 'ሰ': 'ሰ', 'ሸ': 'ሸ', 'ቀ': 'ቀ', 'በ': 'በ',
            'ተ': 'ተ', 'ቸ': 'ቸ', 'ኀ': 'ኀ', 'ነ': 'ነ', 'ኘ': 'ኘ',
            'አ': 'አ', 'ከ': 'ከ', 'ኸ': 'ኸ', 'ወ': 'ወ', 'ዐ': 'ዐ',
            'ዘ': 'ዘ', 'ዠ': 'ዠ', 'የ': 'የ', 'ደ': 'ደ', 'ዸ': 'ዸ',
            'ጀ': 'ጀ', 'ገ': 'ገ', 'ጠ': 'ጠ', 'ጨ': 'ጨ', 'ጰ': 'ጰ',
            'ጸ': 'ጸ', 'ፀ': 'ፀ', 'ፈ': 'ፈ', 'ፐ': 'ፐ'
        }
        
        # Simplified: just return the character itself as the consonant
        return syllable
    
    def _get_vowel(self, syllable: str) -> str:
        """Extract vowel from Ge'ez syllable."""
        # This is simplified - vowel is inherent in the character
        vowel_map = {
            'ሀ': 'ə', 'ሁ': 'u', 'ሂ': 'i', 'ሃ': 'a', 'ሄ': 'e', 'ህ': 'ɨ', 'ሆ': 'o',
            'ለ': 'ə', 'ሉ': 'u', 'ሊ': 'i', 'ላ': 'a', 'ሌ': 'e', 'ል': 'ɨ', 'ሎ': 'o',
            # Add more mappings as needed
        }
        
        return vowel_map.get(syllable, 'ə')