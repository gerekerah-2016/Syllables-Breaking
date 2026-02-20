# test_files.py
import os

print("Current directory:", os.getcwd())
print("\nChecking Geez-Dataset folder:")
print("-" * 50)

source_dir = "Geez-Dataset"
if not os.path.exists(source_dir):
    print(f"❌ {source_dir} does not exist!")
    exit()

# List all folders in Geez-Dataset
folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
print(f"Found {len(folders)} folders:")
for folder in folders:
    print(f"  📁 {folder}")
    
    # Check each folder for .txt files
    folder_path = os.path.join(source_dir, folder)
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    print(f"     {len(txt_files)} .txt files")
    
    # Show first file as example
    if txt_files:
        example_file = os.path.join(folder_path, txt_files[0])
        print(f"     Example: {txt_files[0]}")
        print(f"     Full path: {example_file}")
        
        # Try to read first few lines
        try:
            with open(example_file, 'r', encoding='utf-8') as f:
                lines = [next(f) for _ in range(3)]
                print("     First few lines:")
                for line in lines:
                    print(f"       {line[:50]}...")
        except Exception as e:
            print(f"     Error reading file: {e}")
    
    print()