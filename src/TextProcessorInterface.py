"""
Interface for text processors.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

from abc import ABC, abstractmethod

from src.language_utils.LanguageUtilsInterface import LanguageUtilsInterface


class TextProcessorInterface(ABC):
    """Abstract base class for all text processors."""
    
    def __init__(self, language_utils: LanguageUtilsInterface):
        self.language_utils = language_utils

    @abstractmethod
    def process(self, text: str) -> str:
        """
        Process text according to the specific strategy.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text
        """
        pass