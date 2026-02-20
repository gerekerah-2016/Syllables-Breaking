"""
Fix incomplete splinter mappings.
Run this after training if mappings are incomplete.
"""

import os
import json
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.path_utils import get_splinter_dir, get_experiment_dir
from src.params import set_run_params

def fix_mappings():
    """Fix incomplete mappings like '0:[' → '0:[a]'"""
    
    # Set parameters for the experiment
    set_run_params({
        'OUTPUT_BASE_DIR': './outputs',
        'EXPERIMENT_NAME': '2026-02-17-geez-all_letters',
        'LANGUAGE': 'ge',
        'TASK_ID': 0,
        'TIMESTAMP': '20260217-164753'
    })
    
    print("=" * 60)
    print("FIXING SPLINTER MAPPINGS")
    print("=" * 60)
    
    # Try multiple possible locations for mapping files
    possible_paths = [
        os.path.join(get_splinter_dir(), 'new_unicode_chars.json'),
        os.path.join(get_experiment_dir(), 'results', 'splinter', 'new_unicode_chars.json'),
        './outputs/2026-02-17-geez-all_letters/results/splinter/new_unicode_chars.json'
    ]
    
    encode_path = None
    for path in possible_paths:
        if os.path.exists(path):
            encode_path = path
            print(f"✅ Found mapping file: {path}")
            break
    
    if not encode_path:
        print("❌ No mapping file found!")
        return
    
    # Load existing map
    with open(encode_path, 'r', encoding='utf-8') as f:
        encode_map = json.load(f)
    
    print(f"📊 Loaded {len(encode_map)} mappings")
    
    # Check for incomplete mappings
    fixed_count = 0
    new_encode_map = {}
    
    # Define common vowel patterns based on position
    # These are heuristics - adjust based on your corpus
    position_vowel_map = {
        0: '[a]',   # First consonant often takes [a]
        1: '[u]',   # Second consonant often takes [u]
        2: '[a]',   # Third consonant often takes [a]
        3: '[ə]',   # Fourth consonant often takes [ə]
        4: '[e]',   # Fifth consonant often takes [e]
        5: '[o]',   # Sixth consonant often takes [o]
        6: '[i]',   # Seventh consonant often takes [i]
    }
    
    for key, value in encode_map.items():
        if ':' in key:
            idx, marker = key.split(':', 1)
            
            # If marker is incomplete (just '[' or missing vowel)
            if marker == '[' or marker == '' or len(marker) < 3:
                try:
                    pos = int(idx)
                    # Get appropriate vowel for this position
                    if pos in position_vowel_map:
                        new_marker = position_vowel_map[pos]
                    else:
                        new_marker = '[ə]'  # Default
                    
                    new_key = f"{idx}:{new_marker}"
                    new_encode_map[new_key] = value
                    fixed_count += 1
                    print(f"  Fixed: {key} → {new_key}")
                except ValueError:
                    new_encode_map[key] = value
            else:
                new_encode_map[key] = value
        else:
            new_encode_map[key] = value
    
    # Create inverted map
    decode_map = {v: k for k, v in new_encode_map.items()}
    
    # Save fixed maps
    with open(encode_path, 'w', encoding='utf-8') as f:
        json.dump(new_encode_map, f, indent='\t', ensure_ascii=False)
    
    inv_path = encode_path.replace('new_unicode_chars.json', 'new_unicode_chars_inverted.json')
    with open(inv_path, 'w', encoding='utf-8') as f:
        json.dump(decode_map, f, indent='\t', ensure_ascii=False)
    
    print(f"\n✅ Fixed {fixed_count} mappings")
    print(f"📊 Total mappings now: {len(new_encode_map)}")
    
    # Show sample of fixed mappings
    print("\n📋 Sample mappings:")
    count = 0
    for key, value in new_encode_map.items():
        if ':' in key and len(key) > 3:
            print(f"  {key} → {value}")
            count += 1
            if count >= 10:
                break
    
    print("\n" + "=" * 60)
    print("Next step: Run compare_baseline_splinter.py again")
    print("=" * 60)

if __name__ == "__main__":
    fix_mappings()