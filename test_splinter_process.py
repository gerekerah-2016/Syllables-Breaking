# test_splinter_debug.py
from src.language_utils.EthiopicUtils import EthiopicUtils
from src.SplinterTrainer import SplinterTrainer

utils = EthiopicUtils()
trainer = SplinterTrainer(utils)

test_word = "ባሱማ"
broken = utils.syllable_break(test_word)
print(f"Broken: {broken}")
print(f"Broken as list: {list(broken)}")
print(f"Character codes:")
for i, char in enumerate(broken):
    print(f"  {i}: '{char}' (ord={ord(char)})")

# Process the word
trainer.process_word(test_word)

# Show what's in the encode_map
print(f"\nEncode map ({len(trainer.encode_map)} entries):")
for key, value in trainer.encode_map.items():
    print(f"  {key} -> {value}")