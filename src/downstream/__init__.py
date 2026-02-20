"""
Downstream tasks package for Ethiopic SPLINTER evaluation.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

# Try importing each module with fallback
try:
    from src.downstream.ner import train_ner_with_splinter, load_masakhaner_amharic
    NER_AVAILABLE = True
except (ImportError, Exception) as e:
    NER_AVAILABLE = False
    print(f"⚠️ NER module not available: {e}")

try:
    from src.downstream.pos import train_pos_with_splinter, download_ud_amharic
    POS_AVAILABLE = True
except (ImportError, Exception) as e:
    POS_AVAILABLE = False
    print(f"⚠️ POS module not available: {e}")

try:
    from src.downstream.mt import train_mt_with_splinter, download_jw300_geez_english
    MT_AVAILABLE = True
except (ImportError, Exception) as e:
    MT_AVAILABLE = False
    print(f"⚠️ MT module not available: {e}")

try:
    from src.downstream.classification import train_classification_with_splinter, download_amharic_news
    CLASSIFICATION_AVAILABLE = True
except (ImportError, Exception) as e:
    CLASSIFICATION_AVAILABLE = False
    print(f"⚠️ Classification module not available: {e}")

# Always import run_all if it exists
try:
    from src.downstream.run_all import run_all_downstream_tasks
    RUN_ALL_AVAILABLE = True
except (ImportError, Exception) as e:
    RUN_ALL_AVAILABLE = False
    print(f"⚠️ run_all module not available: {e}")

__all__ = []

if RUN_ALL_AVAILABLE:
    __all__.append('run_all_downstream_tasks')

# Only add to __all__ if available
if NER_AVAILABLE:
    __all__.extend(['train_ner_with_splinter', 'load_masakhaner_amharic'])
if POS_AVAILABLE:
    __all__.extend(['train_pos_with_splinter', 'download_ud_amharic'])
if MT_AVAILABLE:
    __all__.extend(['train_mt_with_splinter', 'download_jw300_geez_english'])
if CLASSIFICATION_AVAILABLE:
    __all__.extend(['train_classification_with_splinter', 'download_amharic_news'])