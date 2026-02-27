# global_decoder.py
import sys
import json
import re
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Import EthiopicUtils directly
from language_utils.EthiopicUtils import EthiopicUtils


class GlobalSplinterDecoder:
    """
    Global Splinter Decoder - FINAL VERSION
    Handles both attached and space-separated formats correctly
    Includes punctuation splitting for perfect bracket handling
    """
    
    def __init__(self, language_utils):
        self.language_utils = language_utils
        self.decode_map = None
        self.cjk_range = (0x4E00, 0x9FFF)
        self.decode_cache = {}
    
    def load_decode_map(self, map_data):
        """Load decode map from dictionary."""
        self.decode_map = map_data
        print(f"✅ Loaded {len(self.decode_map)} mappings")
        return True
    
    def _is_cjk_char(self, char):
        """Check if character is in CJK Unified Ideographs range."""
        if len(char) != 1:
            return False
        return 0x4E00 <= ord(char) <= 0x9FFF
    
    def _reconstruct_word(self, combined_string):
        """
        Reconstruct word using Ghost Index mapping.
        Maps CJK instructions to Ge'ez letters only, ignoring punctuation.
        """
        cache_key = combined_string
        if cache_key in self.decode_cache:
            return self.decode_cache[cache_key]
        
        # Convert to list for modification
        result = list(combined_string)
        
        # 🎯 STEP 1: Find indices of actual Ge'ez letters only
        geez_indices = [
            i for i, char in enumerate(result)
            if self.language_utils.is_letter_in_language(char)
        ]
        
        # Collect all CJK chars and their instructions
        cjk_chars = [c for c in combined_string if self._is_cjk_char(c)]
        
        # 🎯 STEP 2: Parse all instructions from CJK tags
        instructions = []  # (linguistic_position, marker)
        for cjk in cjk_chars:
            if cjk in self.decode_map:
                reduction_key = self.decode_map[cjk]
                parts = reduction_key.split(',')
                for part in parts:
                    part = part.strip()
                    if ':' in part:
                        try:
                            pos_str, marker = part.split(':', 1)
                            instructions.append((int(pos_str), marker.strip()))
                        except ValueError:
                            continue
        
        # Sort instructions by linguistic position
        instructions.sort(key=lambda x: x[0])
        
        # 🎯 STEP 3: Apply vowels using the mapped indices
        modified_geez_positions = set()
        
        for ling_pos, marker in instructions:
            if ling_pos < len(geez_indices) and ling_pos not in modified_geez_positions:
                actual_index = geez_indices[ling_pos]
                base_char = result[actual_index]
                
                order = self.language_utils.get_vowel_order_from_marker(marker)
                if order is not None:
                    result[actual_index] = self.language_utils.apply_vowel_to_consonant(base_char, order)
                    modified_geez_positions.add(ling_pos)
        
        # 🎯 STEP 4: Remove all CJK characters
        final_result = [c for c in result if not self._is_cjk_char(c)]
        reconstructed = ''.join(final_result)
        
        self.decode_cache[cache_key] = reconstructed
        return reconstructed
    
    def decode_line(self, line):
        """
        Decode a line - handles both attached and space-separated formats.
        FINAL VERSION with punctuation splitting for perfect results.
        """
        if not line or not line.strip():
            return line
        
        # 🎯 FIX: Split tokens but keep punctuation separated if they are attached to words
        # This prevents ']' from being part of the 'ረተረ' skeleton.
        line = re.sub(r'([\[\](){},.;:!፡።፣፤፥፦፧፠፨])', r' \1 ', line)
        
        # Remove ▁ markers
        line = line.replace('▁', '')
        
        # Fix number patterns
        line = re.sub(r'(\d+)\s*\(\s*:\s*', r'\1(:', line)
        
        # Split into tokens
        tokens = line.strip().split()
        
        # Result stack - builds output incrementally
        result_stack = []
        i = 0
        n = len(tokens)
        
        # Track if this line has CJK for debugging
        has_cjk = any(self._is_cjk_char(c) for token in tokens for c in token)
        if has_cjk:
            print(f"\n{'='*60}")
            print(f">>> DECODER PROCESSING: '{line}'")
            print(f">>> Tokens: {tokens}")
            print(f"{'='*60}")
        
        while i < n:
            current = tokens[i]
            
            # Check if current token contains CJK
            has_cjk_here = any(self._is_cjk_char(c) for c in current)
            
            if has_cjk_here:
                # Check if skeleton is in this token (attached case)
                contains_geez = any(self.language_utils.is_letter_in_language(c) for c in current)
                
                if contains_geez:
                    # ATTACHED CASE: e.g., 'ለየ三七' or '1(:አረተ丝丏丁'
                    reconstructed = self._reconstruct_word(current)
                    result_stack.append(reconstructed)
                    if has_cjk:
                        print(f">>> Attached token '{current}' → '{reconstructed}'")
                    i += 1
                else:
                    # DISCONNECTED CASE: e.g., 'አ ረ ተ' then '丝丏丁'
                    # Collect all consecutive CJK tokens
                    cjk_tokens = [current]
                    i += 1
                    while i < n and any(self._is_cjk_char(c) for c in tokens[i]):
                        cjk_tokens.append(tokens[i])
                        i += 1
                    cjk_string = ''.join(cjk_tokens)
                    
                    # Look backwards to find Ge'ez tokens that form the skeleton
                    skeleton_tokens = []
                    j = len(result_stack) - 1
                    while j >= 0:
                        stack_item = result_stack[j]
                        if any(self.language_utils.is_letter_in_language(c) for c in stack_item):
                            skeleton_tokens.insert(0, result_stack.pop(j))
                            j -= 1
                        else:
                            # Stop at first non-Ge'ez (punctuation)
                            break
                    
                    if skeleton_tokens:
                        # Combine skeleton tokens with CJK
                        combined = ''.join(skeleton_tokens) + cjk_string
                        reconstructed = self._reconstruct_word(combined)
                        result_stack.append(reconstructed)
                        if has_cjk:
                            print(f">>> Reconstructed '{combined}' → '{reconstructed}'")
                    else:
                        # Fallback: keep non-CJK parts
                        non_cjk = ''.join([c for c in cjk_string if not self._is_cjk_char(c)])
                        if non_cjk:
                            result_stack.append(non_cjk)
            else:
                # Regular word or punctuation
                result_stack.append(current)
                i += 1
        
        # Join stack with spaces
        final_output = ' '.join(result_stack)
        
        # 🧼 Final cleanup: Remove structural markers and normalize
        final_output = final_output.replace('▁', '')
        final_output = re.sub(r'\s+([፡።፣፤፥፦፧፠፨\(\)\[\]\{\}.,;:!?])', r'\1', final_output)
        final_output = re.sub(r'([፡።፣፤፥፦፧፠፨\(\)\[\]\{\}.,;:!?])\s+', r'\1', final_output)
        final_output = re.sub(r'\(\s*:', r'(:', final_output)
        final_output = re.sub(r':\s*\)', r':)', final_output)
        final_output = re.sub(r'\s+', ' ', final_output)
        
        return final_output.strip()


def main():
    print("="*60)
    print("GLOBAL DECODER - FINAL VERSION")
    print("="*60)
    
    # Initialize
    print("\n1. Initializing EthiopicUtils...")
    utils = EthiopicUtils()
    
    # Create decoder
    print("\n2. Creating decoder...")
    decoder = GlobalSplinterDecoder(utils)
    
    # Your actual mapping
    decode_map = {
        "一": "3:[ï]",
        "丁": "2:[ï]",
        "丂": "4:[ï]",
        "七": "1:[ï]",
        "丄": "5:[ï]",
        "丅": "0:[ï]",
        "丆": "2:[a]",
        "万": "1:[a]",
        "丈": "3:[a]",
        "三": "0:[a]",
        "上": "6:[ï]",
        "下": "2:[o]",
        "丌": "3:[u]",
        "不": "4:[a]",
        "与": "3:[o]",
        "丏": "1:[i]",
        "丐": "4:[u]",
        "丑": "1:[o]",
        "丒": "2:[u]",
        "专": "2:[i]",
        "且": "0:[u]",
        "丕": "5:[a]",
        "世": "1:[u]",
        "丗": "3:[i]",
        "丘": "0:[i]",
        "丙": "0:[e]",
        "业": "7:[ï]",
        "丛": "4:[o]",
        "东": "1:[e]",
        "丝": "0:[o]",
        "丞": "2:[e]",
        "丟": "5:[u]",
        "丠": "4:[i]",
        "両": "3:[e]",
        "丢": "5:[i]",
        "丣": "1:[9]",
        "两": "8:[ï]",
        "严": "2:[7]",
        "並": "6:[a]",
        "丧": "6:[u]",
        "丨": "5:[o]",
        "丩": "3:[7]",
        "个": "0:[9]",
        "丫": "2:[9]",
        "丬": "3:[9]",
        "中": "4:[e]",
        "丮": "5:[e]",
        "丯": "1:[7]",
        "丰": "4:[7]",
        "丱": "4:[9]",
        "串": "6:[i]",
        "丳": "6:[o]",
        "临": "7:[a]",
        "丵": "9:[ï]",
        "丶": "7:[u]",
        "丷": "8:[a]",
        "丸": "10:[ï]",
        "丹": "0:[7]",
        "为": "0:[11]",
        "主": "8:[u]",
        "丼": "7:[o]",
        "丽": "7:[i]",
        "举": "5:[7]",
        "丿": "6:[e]",
        "乀": "9:[a]",
        "乁": "1:[11]",
        "乂": "8:[i]",
        "乃": "8:[o]",
        "乄": "10:[a]",
        "久": "9:[u]",
        "乆": "11:[ï]",
        "乇": "9:[i]",
        "么": "9:[o]",
        "义": "10:[u]",
        "乊": "11:[a]",
        "之": "5:[9]",
        "乌": "8:[e]",
        "乍": "7:[e]",
        "乎": "2:[11]",
        "乏": "10:[i]",
        "乐": "12:[ï]",
        "乑": "6:[7]",
        "乒": "10:[o]",
        "乓": "12:[a]",
        "乔": "9:[e]",
        "乕": "11:[u]",
        "乖": "7:[7]",
        "乗": "3:[11]",
        "乘": "11:[i]",
        "乙": "13:[ï]",
        "乚": "14:[ï]",
        "乛": "6:[9]",
        "乜": "8:[7]",
        "九": "10:[e]",
        "乞": "11:[o]",
        "也": "12:[u]",
        "习": "4:[11]",
        "乡": "9:[7]",
        "乢": "2:[8]",
        "乣": "13:[a]",
        "乤": "15:[ï]",
        "乥": "16:[ï]",
        "书": "0:[8]",
        "乧": "11:[e]",
        "乨": "12:[o]",
        "乩": "0:[ï],1:[ï],2:[ï]",
        "乪": "0:[a],1:[ï]",
        "乫": "0:[ï],1:[ï]",
        "乬": "1:[ï],2:[ï]",
        "乭": "0:[ï],1:[a]",
        "乮": "1:[a],2:[ï]",
        "乯": "1:[i],2:[ï]",
        "买": "1:[ï],2:[o]",
        "乱": "0:[a],2:[ï]",
        "乲": "0:[ï],1:[ï],2:[ï],3:[ï]",
        "乳": "1:[ï],3:[ï]",
        "乴": "0:[ï],1:[a],2:[ï]",
        "乵": "2:[a],3:[ï]",
        "乶": "1:[ï],2:[ï],3:[ï]",
        "乷": "1:[ï],2:[a],3:[ï]",
    }
    
    decoder.load_decode_map(decode_map)
    
    # Test cases
    test_cases = [
        # Attached format (no spaces)
        ("1(:አረተ丝丏丁", "1(:ኦሪት"),
        ("ዘደገመ万丁一", "ዘዳግም"),
        ("።)ዘነተ丅七丒", "።)ዝንቱ"),
        ("ወአተ丅七丒", "ውእቱ"),
        ("ነገረ丁", "ነገር"),
        ("ዘነገረመ与丐", "ዘነገሮሙ"),
        ("መሰ且东", "ሙሴ"),
        ("[ አ 丙 ]ረተረ丅七丆", "[ኤ]ርትራ"),
        
        # Space-separated format
        ("1 ( : አ ረ ተ 丝 丏丁", "1(:ኦሪት"),
        ("ዘ ደ ገ መ 万 丁 一", "ዘዳግም"),
        ("። ) ዘ ነ ተ 丅 七 丒", "።)ዝንቱ"),
        ("ወ አ ተ 丅 七 丒", "ውእቱ"),
        ("ነ ገ ረ 丁", "ነገር"),
        ("[ አ 丙 ] ረተ ረ 丅七丆", "[ኤ]ርትራ"),
        
        # Problematic attached cases
        ("ለየ三七", "ላይ"),
        ("አሰከረነ七両丂", "አስከሬን"),
        ("በለፈወ三一", "ባለፈው"),
        ("ቀየተወለ丝七丈丂፡፡", "ቆይተዋል፡፡"),
        ("ከሰመነተ万丁一丂", "ከሳምንት"),
        ("የአለተ七丌", "የእለቱ"),
    ]
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for i, (input_text, expected) in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"RUNNING TEST {i}")
        print(f"{'='*60}")
        result = decoder.decode_line(input_text)
        status = "✅ PASSED" if result == expected else "❌ FAILED"
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{i}. Input:  '{input_text}'")
        print(f"   Result: '{result}'")
        print(f"   Expect: '{expected}'")
        print(f"   {status}")
        
        if result != expected:
            # Show what went wrong
            if result.replace(' ', '') == expected.replace(' ', ''):
                print(f"      Issue: Extra/missing spaces")
            else:
                # Show character by character
                for j, (r_char, e_char) in enumerate(zip(result, expected)):
                    if r_char != e_char:
                        print(f"      First diff at position {j}: '{r_char}' vs '{e_char}'")
                        break
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()