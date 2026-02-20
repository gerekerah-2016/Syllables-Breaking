import json
import os
from src.utils.path_utils import get_splinter_dir

# Load existing map
splinter_dir = get_splinter_dir()
map_path = os.path.join(splinter_dir, 'new_unicode_chars.json')

if os.path.exists(map_path):
    with open(map_path, 'r', encoding='utf-8') as f:
        encode_map = json.load(f)
else:
    encode_map = {}

# Fix the mappings - these should match your example
correct_mappings = {
    '0:[a]': '⟨1⟩',
    '1:[u]': '⟨2⟩', 
    '2:[a]': '⟨3⟩',
    '3:[ə]': '⟨4⟩',
    '4:[e]': '⟨5⟩',
    '5:[o]': '⟨6⟩',
    '6:[i]': '⟨7⟩',
    '7:[ɨ]': '⟨8⟩',
}

# Update with correct mappings
encode_map.update(correct_mappings)

# Save back
with open(map_path, 'w', encoding='utf-8') as f:
    json.dump(encode_map, f, indent='\t', ensure_ascii=False)

# Create inverted map
decode_map = {v: k for k, v in encode_map.items()}
inv_path = os.path.join(splinter_dir, 'new_unicode_chars_inverted.json')
with open(inv_path, 'w', encoding='utf-8') as f:
    json.dump(decode_map, f, indent='\t', ensure_ascii=False)

print("✅ Fixed mappings:")
for k, v in correct_mappings.items():
    print(f"  {k} → {v}")