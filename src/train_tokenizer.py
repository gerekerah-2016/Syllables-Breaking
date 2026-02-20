import os
import json
import sentencepiece as spm
from src.logger import get_logger
from src.params import get_run_params
from src.utils.path_utils import get_logs_dir, get_splinter_dir
from src.utils.utils import decode_tokens_vocab_file

def train_tokenizer(tokenizer_type, vocab_size, input_path, output_path):
    log_path = f"{get_logs_dir()}/SentencePieceTrainer - {tokenizer_type}_{vocab_size}.log"
    logger = get_logger()
    
    logger.info(f'Start training Ge\'ez tokenizer {tokenizer_type}_{vocab_size}.')
    
    # 1. Load Special Symbols (⟨n⟩) to ensure they aren't split
    special_symbols = []
    splinter_dir = get_splinter_dir()
    maps_path = os.path.join(splinter_dir, 'new_unicode_chars.json')
    
    if os.path.exists(maps_path):
        with open(maps_path, 'r', encoding='utf-8') as f:
            maps = json.load(f)
            special_symbols = [v for v in maps.values() if v.startswith('⟨') and v.endswith('⟩')]
    
    # 2. Execute Training
    _train_tokenizer(
        input_file=input_path,
        output_path=output_path,
        tokenizer_type=tokenizer_type,
        vocab_size=vocab_size,
        log_path=log_path,
        special_symbols=special_symbols
    )
    
    logger.info(f'Finished training. Logs: {log_path}')

    if get_run_params("IS_ENCODED"):
        decode_tokens_vocab_file(output_path)


def _train_tokenizer(input_file, output_path, tokenizer_type, vocab_size, log_path, special_symbols):
    # Convert list to comma-separated string for SentencePiece
    symbols_str = ','.join(special_symbols) if special_symbols else ''
    
    with open(log_path, 'w', encoding='utf-8') as log_file:
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=output_path,
            vocab_size=vocab_size,
            model_type=tokenizer_type,
            logstream=log_file,
            
            # --- GE'EZ SPECIFIC OPTIMIZATIONS ---
            character_coverage=1.0,          # Essential: Don't turn rare Fidels into <unk>
            user_defined_symbols=symbols_str, # Protects your ⟨n⟩ tags from being broken
            split_by_unicode_script=True,    # Prevents Ge'ez/Latin script mixing in tokens
            split_by_whitespace=True,        # Respects word boundaries
            
            # Standard IDs
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>'
        )