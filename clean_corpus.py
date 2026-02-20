# clean_corpus.py
import os
import re
from pathlib import Path

def is_ethiopic(text):
    """Check if text is primarily Ethiopic."""
    if not text:
        return False
    
    ethiopic_count = 0
    total_count = 0
    
    for char in text:
        if 0x1200 <= ord(char) <= 0x137F:  # Ethiopic range
            ethiopic_count += 1
            total_count += 1
        elif char.isspace() or char in '.,:;!?()-':
            total_count += 1
    
    if total_count == 0:
        return False
    
    return (ethiopic_count / total_count) > 0.5  # At least 50% Ethiopic

def clean_file(input_path, output_path):
    """Clean a file by removing non-Ethiopic content."""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into lines
    lines = content.split('\n')
    
    # Keep only lines with Ethiopic content
    ethiopic_lines = [line for line in lines if is_ethiopic(line)]
    
    if ethiopic_lines:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(ethiopic_lines))
        return True
    return False

def main():
    # Create clean directory
    clean_dir = Path("./Geez-Dataset-Clean")
    clean_dir.mkdir(exist_ok=True)
    
    # Process all txt files
    data_dir = Path("./Geez-Dataset")
    txt_files = list(data_dir.glob("*.txt"))
    
    print(f"Found {len(txt_files)} files to clean")
    
    cleaned_count = 0
    for file in txt_files:
        output_path = clean_dir / file.name
        if clean_file(file, output_path):
            cleaned_count += 1
            print(f"✅ Cleaned: {file.name}")
        else:
            print(f"❌ No Ethiopic content: {file.name}")
    
    print(f"\nCleaned {cleaned_count} files to {clean_dir}")

if __name__ == "__main__":
    main()