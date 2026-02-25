"""
Pre-train BERT on Ge'ez/Amharic corpus
"""

import os
from pathlib import Path
from transformers import (
    BertConfig, BertForMaskedLM, BertTokenizerFast,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeEzBertPreTrainer:
    """Pre-train BERT on Ge'ez/Amharic corpus"""
    
    def __init__(self, output_dir="./geez_bert"):
        self.logger = logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_tokenizer(self, corpus_file, vocab_size=32000):
        """Train a new tokenizer on Ge'ez corpus"""
        from tokenizers import BertWordPieceTokenizer
        
        self.logger.info(f"Training tokenizer with vocab size {vocab_size}...")
        
        tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=False,
        )
        
        # Train tokenizer
        tokenizer.train(
            files=[corpus_file],
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        )
        
        # Save tokenizer
        tokenizer.save_model(str(self.output_dir / "tokenizer"))
        self.logger.info(f"Tokenizer saved to {self.output_dir / 'tokenizer'}")
        
        # Load as HuggingFace tokenizer
        hf_tokenizer = BertTokenizerFast.from_pretrained(str(self.output_dir / "tokenizer"))
        hf_tokenizer.save_pretrained(str(self.output_dir / "tokenizer"))
        
        return hf_tokenizer
    
    def prepare_dataset(self, corpus_file, tokenizer, max_length=128, sample_size=10000):
        """Prepare dataset for MLM training"""
        
        # Load text file (sample for speed)
        dataset = load_dataset('text', data_files=corpus_file, split='train')
        
        # Sample if too large
        if len(dataset) > sample_size:
            dataset = dataset.select(range(sample_size))
            self.logger.info(f"Sampled {sample_size} lines for training")
        
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_special_tokens_mask=True
            )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text'],
            num_proc=1
        )
        
        return tokenized_dataset
    
    def pretrain(self, corpus_file, vocab_size=8000, num_epochs=1):
        """Run pre-training with smaller config for testing"""
        
        # Step 1: Create tokenizer
        tokenizer = self.create_tokenizer(corpus_file, vocab_size)
        
        # Step 2: Prepare dataset
        train_dataset = self.prepare_dataset(corpus_file, tokenizer)
        
        # Step 3: Initialize model (small config for testing)
        config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=256,  # Smaller for testing
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            max_position_embeddings=512,
        )
        
        model = BertForMaskedLM(config=config)
        self.logger.info(f"Model initialized with {model.num_parameters():,} parameters")
        
        # Step 4: Data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
        
        # Step 5: Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=8,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=100,
        )
        
        # Step 6: Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        
        # Step 7: Train!
        self.logger.info("Starting pre-training...")
        trainer.train()
        
        # Step 8: Save final model
        model.save_pretrained(str(self.output_dir / "final_model"))
        tokenizer.save_pretrained(str(self.output_dir / "final_model"))
        
        self.logger.info(f"Pre-training complete! Model saved to {self.output_dir / 'final_model'}")
        
        return model, tokenizer
    
    def pretrain_with_small_config(self, corpus_file, vocab_size=8000, num_epochs=1):
        """Alias for pretrain method"""
        return self.pretrain(corpus_file, vocab_size, num_epochs)


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    pretrainer = GeEzBertPreTrainer(output_dir="./geez_bert_pretrained")
    
    # Check if corpus exists
    corpus_file = "./pretrain_corpus.txt"
    if not os.path.exists(corpus_file):
        print(f"❌ Corpus file {corpus_file} not found!")
        print("   Run unified_pos_data.py first to create it.")
        sys.exit(1)
    
    # Pre-train with small config for testing
    model, tokenizer = pretrainer.pretrain_with_small_config(
        corpus_file=corpus_file,
        vocab_size=8000,
        num_epochs=1
    )
    print("✅ Pre-training complete!")