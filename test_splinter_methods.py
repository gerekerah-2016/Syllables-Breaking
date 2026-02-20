# test_splinter_methods.py
from src.language_utils.EthiopicUtils import EthiopicUtils
from src.SplinterTrainer import SplinterTrainer
import inspect

utils = EthiopicUtils()
trainer = SplinterTrainer(utils)

print("Methods in SplinterTrainer:")
for name, method in inspect.getmembers(trainer, inspect.ismethod):
    print(f"  - {name}")