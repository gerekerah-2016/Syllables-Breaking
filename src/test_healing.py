# test_healing_standalone.py
import sys
import json
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import EthiopicUtils directly
from language_utils.EthiopicUtils import EthiopicUtils

# Create a simplified decoder for testing (no logger dependencies)
class SimpleDecoder:
    def __init__(self, language_utils):
        self.language_utils = language_utils
        self.decode_map = None
        self.cjk_range = (0x4E00, 0x9FFF)
    
    def _is_cjk_char(self, char):
        if len(char) != 1:
            return False
        return 0x4E00 <= ord(char) <= 0x9FFF
    
    def _heal_line(self, line):
        # Remove spaces between Ge'ez and CJK
        line = re.sub(r'([\u1200-\u137F])\s+([\u4E00-\u9FFF])', r'\1\2', line)
        # Remove spaces between CJK characters
        line = re.sub(r'([\u4E00-\u9FFF])\s+([\u4E00-\u9FFF])', r'\1\2', line)
        # Fix number patterns
        line = re.sub(r'(\d+)\s*\(\s*:\s*', r'\1(:', line)
        return line
    
    def decode_line(self, line):
        if not line:
            return line
        
        # Remove SentencePiece markers
        line = line.replace('▁', '')
        
        # HEALING PHASE
        line = self._heal_line(line)
        
        # Split into parts
        parts = re.findall(r'[\u1200-\u137F]+|[\u4E00-\u9FFF]+|[^\u1200-\u137F\u4E00-\u9FFF]+', line)
        
        decoded_parts = []
        last_geez = ""
        
        for part in parts:
            if not part:
                continue
            
            first_char = part[0]
            
            if self.language_utils.is_letter_in_language(first_char):
                if last_geez:
                    decoded_parts.append(last_geez)
                last_geez = part
            
            elif self._is_cjk_char(first_char):
                if last_geez and self.decode_map:
                    reduction_keys = []
                    for c in part:
                        if c in self.decode_map:
                            reduction_keys.append(self.decode_map[c])
                    
                    if reduction_keys and hasattr(self.language_utils, 'decode_from_splinter_with_keys'):
                        decoded = self.language_utils.decode_from_splinter_with_keys(
                            last_geez, reduction_keys, self.decode_map
                        )
                        decoded_parts.append(decoded)
                    else:
                        decoded_parts.append(last_geez)
                    last_geez = ""
                else:
                    decoded_parts.append(part)
            
            else:
                if last_geez:
                    decoded_parts.append(last_geez)
                    last_geez = ""
                decoded_parts.append(part)
        
        if last_geez:
            decoded_parts.append(last_geez)
        
        return ''.join(decoded_parts).strip()

def main():
    print("="*60)
    print("STANDALONE DECODER TEST WITH HEALING")
    print("="*60)
    
    # Initialize
    utils = EthiopicUtils()
    decoder = SimpleDecoder(utils)
    
    # Load decode map
    map_path = Path("../outputs/2026-02-17-geez-all_letters/results/splinter/new_unicode_chars_inverted.json")
    if not map_path.exists():
        print(f"❌ Map not found at {map_path}")
        # Try alternative path
        map_path = Path("outputs/2026-02-17-geez-all_letters/results/splinter/new_unicode_chars_inverted.json")
        if not map_path.exists():
            print(f"❌ Map also not found at {map_path}")
            return
    
    with open(map_path, 'r', encoding='utf-8') as f:
        decoder.decode_map = json.load(f)
    print(f"✅ Loaded {len(decoder.decode_map)} mappings")
    
    # Show sample mappings
    print("\nSample mappings:")
    for cjk, val in list(decoder.decode_map.items())[:5]:
        print(f"  '{cjk}' → '{val}'")
    
    # Test cases
    test_cases = [
        "አረተ 丝丏丁",
        "መወዐለ万丁一",
        "1 ( : አረተ 业 与 一",
        "ዘደገመ 万一丁",
        "። ) ዘነተ 七 丂 不",
        "ወአተ 七 丂 不",
        "ነገረ 一",
        "ዘነገረመ 丐 且",
    ]
    
    expected = [
        "ኦሪት",
        "መዋዕል",
        "1(:ኦሪት",
        "ዘዳግም",
        "።)ዝንቱ",
        "ውእቱ",
        "ነገረ",
        "ዘነገሮሙ",
    ]
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    for i, (test, exp) in enumerate(zip(test_cases, expected), 1):
        print(f"\n{i}. Original: '{test}'")
        
        # Show healing
        healed = decoder._heal_line(test)
        print(f"   Healed:   '{healed}'")
        
        # Decode
        result = decoder.decode_line(test)
        print(f"   Result:   '{result}'")
        print(f"   Expected: '{exp}'")
        
        if result == exp:
            print(f"   ✅ SUCCESS")
        else:
            print(f"   ❌ FAILED")
            
            # Debug
            print(f"   Debug - CJK in input:")
            for c in test:
                if decoder._is_cjk_char(c):
                    if c in decoder.decode_map:
                        print(f"     '{c}' → '{decoder.decode_map[c]}'")
                    else:
                        print(f"     '{c}' → NOT FOUND")

if __name__ == "__main__":
    main()