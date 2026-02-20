"""
Amharic language utilities.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

from src.language_utils.GeezUtils import GeezUtils


class AmharicUtils(GeezUtils):
    """Amharic language utilities."""
    
    def __init__(self):
        super().__init__()
        self.name = "Amharic"
        self.code = "am"
        
        # Amharic additional characters
        self.AMHARIC_EXTRA = ['ኧ', 'ኡ', 'ኢ', 'ኣ', 'ኤ', 'እ', 'ኦ']
        
        # Labiovelar variants common in Amharic
        self.LABIOVELARS = ['ቈ', 'ቊ', 'ቋ', 'ቌ', 'ቍ']
    
    def normalize_text(self, text: str) -> str:
        """Normalize Amharic text."""
        normalized = super().normalize_text(text)
        
        # Amharic-specific normalizations
        # Handle labiovelars
        labiovelar_map = {
            'ቈ': 'ቀው', 'ቊ': 'ቁው', 'ቋ': 'ቃው', 'ቌ': 'ቄው', 'ቍ': 'ቅው'
        }
        
        for old, new in labiovelar_map.items():
            normalized = normalized.replace(old, new)
        
        return normalized