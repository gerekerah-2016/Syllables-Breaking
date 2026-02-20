"""
Tigrinya language utilities.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

from src.language_utils.GeezUtils import GeezUtils


class TigrinyaUtils(GeezUtils):
    """Tigrinya language utilities."""
    
    def __init__(self):
        super().__init__()
        self.name = "Tigrinya"
        self.code = "ti"
        
        # Tigrinya specific characters
        self.TIGRINYA_SPECIAL = ['ቐ', 'ቑ', 'ቒ', 'ቓ', 'ቔ', 'ቕ']
    
    def normalize_text(self, text: str) -> str:
        """Normalize Tigrinya text."""
        normalized = super().normalize_text(text)
        
        # Tigrinya-specific normalizations
        # Handle special velar variants
        velar_map = {
            'ቐ': 'ከ', 'ቑ': 'ኩ', 'ቒ': 'ኪ', 'ቓ': 'ካ', 'ቔ': 'ኬ', 'ቕ': 'ክ'
        }
        
        for old, new in velar_map.items():
            normalized = normalized.replace(old, new)
        
        return normalized