"""
Map SPLINTER tags to single Unicode symbols.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import json
from pathlib import Path

def create_tag_mapping(output_path="tag_mapping.json"):
    """
    Create a mapping from tag numbers to single Unicode symbols.
    Uses Mathematical Alphanumeric Symbols (U+1D400 to U+1D7FF)
    which are safe and displayable.
    """
    
    # Start from Mathematical Bold Capital A (U+1D400)
    START_CODEPOINT = 0x1D400
    
    # We have 83 tags (1-83)
    mapping = {}
    reverse_mapping = {}
    
    for i in range(1, 84):
        # Calculate Unicode codepoint
        codepoint = START_CODEPOINT + i
        symbol = chr(codepoint)
        
        # Store mappings
        mapping[str(i)] = symbol
        reverse_mapping[symbol] = str(i)
    
    # Save mappings
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'tag_to_symbol': mapping,
            'symbol_to_tag': reverse_mapping
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created mapping for 83 tags")
    print(f"   Example: ⟨1⟩ → {mapping['1']}")
    print(f"   Example: ⟨7⟩ → {mapping['7']}")
    print(f"   Example: ⟨16⟩ → {mapping['16']}")
    print(f"   Example: ⟨20⟩ → {mapping['20']}")
    print(f"   Example: ⟨83⟩ → {mapping['83']}")
    
    return mapping, reverse_mapping

def update_text_processor():
    """
    Update TextProcessorWithEncoding.py to use single symbols.
    """
    # The key changes needed:
    print("\n" + "="*60)
    print("CHANGES NEEDED IN TextProcessorWithEncoding.py:")
    print("="*60)
    print("""
    # 1. Load the tag mapping
    with open('tag_mapping.json', 'r', encoding='utf-8') as f:
        tag_mapping = json.load(f)
        self.tag_to_symbol = tag_mapping['tag_to_symbol']
        self.symbol_to_tag = tag_mapping['symbol_to_tag']
    
    # 2. In encode_word, replace this:
    tag = f"⟨{tag_num}⟩"
    
    # With this:
    tag = self.tag_to_symbol[str(tag_num)]
    
    # 3. In decode_word, replace this:
    tags = re.findall(r'⟨\d+⟩', normalized)
    
    # With this:
    # Extract symbols (Mathematical Alphanumeric range)
    symbol_tags = re.findall(r'[\uD400-\uD7FF]', normalized)
    for sym in symbol_tags:
        tag_num = self.symbol_to_tag[sym]
        tags.append(f"⟨{tag_num}⟩")
    """)

def update_train_tokenizer():
    """
    Update train_tokenizer.py settings.
    """
    print("\n" + "="*60)
    print("CHANGES NEEDED IN train_tokenizer.py:")
    print("="*60)
    print("""
    # In the splintered mode section, use these settings:
    
    # CRITICAL: Mathematical symbols are normal Unicode
    character_coverage = 1.0
    byte_fallback = False  # NO byte fallback!
    split_digits = False
    
    # NO special symbols needed - they're normal Unicode!
    # Remove user_defined_symbols completely
    """)

def update_decode_utils():
    """
    Update decode_utils.py to handle symbols.
    """
    print("\n" + "="*60)
    print("CHANGES NEEDED IN decode_utils.py:")
    print("="*60)
    print("""
    # Add this method:
    def _normalize_symbols(self, text: str) -> str:
        \"\"\"
        Convert Mathematical Alphanumeric symbols back to ⟨n⟩.
        \"\"\"
        # Load mapping (do this in __init__)
        if not hasattr(self, 'symbol_to_tag'):
            with open('tag_mapping.json', 'r', encoding='utf-8') as f:
                tag_mapping = json.load(f)
                self.symbol_to_tag = tag_mapping['symbol_to_tag']
        
        # Replace symbols with ⟨n⟩
        for sym, tag_num in self.symbol_to_tag.items():
            text = text.replace(sym, f"⟨{tag_num}⟩")
        
        return text
    
    # Then in decode_line, call this BEFORE _normalize_tags
    text = self._normalize_symbols(text)
    """)

if __name__ == "__main__":
    # Create the mapping
    mapping, reverse = create_tag_mapping()
    
    # Show instructions
    update_text_processor()
    update_train_tokenizer()
    update_decode_utils()