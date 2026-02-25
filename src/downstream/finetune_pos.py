"""
Fine-tune BERT on POS tagging for Amharic/Ge'ez
"""

import os
import torch
import numpy as np
from pathlib import Path
from transformers import (
    BertForTokenClassification, BertTokenizerFast,
    Trainer, TrainingArguments, DataCollatorForTokenClassification
)
from datasets import Dataset
import logging
import conllu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeEzPOSFineTuner:
    """Fine-tune pre-trained BERT on POS tagging"""
    
    def __init__(self, pretrained_model_path="bert-base-multilingual-cased"):
        self.logger = logger
        self.pretrained_path = pretrained_model_path
        
        # Universal POS tags
        self.label_list = [
            'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
            'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'
        ]
        self.id2label = {i: tag for i, tag in enumerate(self.label_list)}
        self.label2id = {tag: i for i, tag in enumerate(self.label_list)}
    
    def load_model_and_tokenizer(self):
        """Load pre-trained model and tokenizer"""
        
        self.logger.info(f"Loading model from {self.pretrained_path}")
        
        # Load tokenizer
        try:
            tokenizer = BertTokenizerFast.from_pretrained(self.pretrained_path)
        except:
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
        
        # Load model for token classification
        try:
            model = BertForTokenClassification.from_pretrained(
                self.pretrained_path,
                num_labels=len(self.label_list),
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True
            )
        except:
            model = BertForTokenClassification.from_pretrained(
                "bert-base-multilingual-cased",
                num_labels=len(self.label_list),
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True
            )
        
        return model, tokenizer
    
    def load_conllu_file(self, file_path):
        """Load sentences from CONLL-U file"""
        sentences = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = f.read()
                sentences = list(conllu.parse(data))
            self.logger.info(f"Loaded {len(sentences)} sentences from {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
        return sentences
    
    def prepare_dataset_from_files(self, train_file, dev_file=None, test_file=None):
        """Prepare datasets from CONLL-U files"""
        
        # Load sentences
        train_sentences = self.load_conllu_file(train_file)
        dev_sentences = self.load_conllu_file(dev_file) if dev_file and os.path.exists(dev_file) else []
        test_sentences = self.load_conllu_file(test_file) if test_file and os.path.exists(test_file) else []
        
        # Load model and tokenizer first
        model, tokenizer = self.load_model_and_tokenizer()
        
        return self.prepare_dataset(train_sentences, dev_sentences, test_sentences, tokenizer)
    
    def prepare_dataset(self, train_sentences, dev_sentences, test_sentences, tokenizer):
        """Convert CONLL-U sentences to tokenized dataset"""
        
        def extract_tokens_and_tags(sentences):
            all_tokens = []
            all_tags = []
            
            for sent in sentences:
                tokens = []
                tags = []
                for token in sent:
                    if isinstance(token, dict) and 'form' in token and 'upos' in token:
                        tokens.append(token['form'])
                        tags.append(token['upos'])
                    elif hasattr(token, 'form') and hasattr(token, 'upos'):
                        tokens.append(token.form)
                        tags.append(token.upos)
                
                if tokens:  # Only add if we have tokens
                    all_tokens.append(tokens)
                    all_tags.append(tags)
            
            return all_tokens, all_tags
        
        # Prepare datasets
        train_tokens, train_tags = extract_tokens_and_tags(train_sentences)
        
        if not train_tokens:
            self.logger.error("No training data available!")
            return None, None, None
        
        train_dataset = Dataset.from_dict({
            'tokens': train_tokens,
            'pos_tags': train_tags
        })
        
        # Tokenization function - now using tokenizer from outer scope
        def tokenize_and_align_labels(examples):
            tokenized = tokenizer(
                examples['tokens'],
                truncation=True,
                padding='max_length',
                max_length=128,
                is_split_into_words=True
            )
            
            labels = []
            for i, label in enumerate(examples['pos_tags']):
                word_ids = tokenized.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        if word_idx < len(label):
                            tag = label[word_idx]
                            label_ids.append(self.label2id.get(tag, 0))
                        else:
                            label_ids.append(-100)
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                
                labels.append(label_ids)
            
            tokenized['labels'] = labels
            return tokenized
        
        # Tokenize training data
        self.logger.info("Tokenizing training data...")
        train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
        
        # Prepare dev dataset if available
        dev_dataset = None
        if dev_sentences:
            dev_tokens, dev_tags = extract_tokens_and_tags(dev_sentences)
            if dev_tokens:
                dev_dataset = Dataset.from_dict({
                    'tokens': dev_tokens,
                    'pos_tags': dev_tags
                })
                self.logger.info("Tokenizing dev data...")
                dev_dataset = dev_dataset.map(tokenize_and_align_labels, batched=True)
        
        # Prepare test dataset if available
        test_dataset = None
        if test_sentences:
            test_tokens, test_tags = extract_tokens_and_tags(test_sentences)
            if test_tokens:
                test_dataset = Dataset.from_dict({
                    'tokens': test_tokens,
                    'pos_tags': test_tags
                })
                self.logger.info("Tokenizing test data...")
                test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)
        
        return train_dataset, dev_dataset, test_dataset
    
    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            pred_seq = []
            label_seq = []
            for p, l in zip(prediction, label):
                if l != -100:
                    pred_seq.append(self.id2label.get(p, 'X'))
                    label_seq.append(self.id2label.get(l, 'X'))
            if pred_seq:
                true_predictions.append(pred_seq)
                true_labels.append(label_seq)
        
        if not true_predictions:
            return {"accuracy": 0.0, "f1": 0.0}
        
        # Calculate accuracy
        correct = 0
        total = 0
        for pred_seq, label_seq in zip(true_predictions, true_labels):
            for p, l in zip(pred_seq, label_seq):
                if p == l:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        # Calculate F1
        try:
            from sklearn.metrics import f1_score
            flat_pred = [tag for seq in true_predictions for tag in seq]
            flat_label = [tag for seq in true_labels for tag in seq]
            f1 = f1_score(flat_label, flat_pred, average='macro')
        except:
            f1 = accuracy
        
        return {"accuracy": accuracy, "f1": f1}
    
    def finetune(self, train_file=None, dev_file=None, test_file=None, 
                 output_dir="./finetuned_pos", num_epochs=3):
        """Fine-tune on POS task using CONLL-U files"""
        
        # Use default files if not provided
        if train_file is None:
            train_file = "./combined_pos_data/train.conllu"
            dev_file = "./combined_pos_data/dev.conllu"
            test_file = "./combined_pos_data/test.conllu"
        
        # Prepare datasets
        train_dataset, dev_dataset, test_dataset = self.prepare_dataset_from_files(
            train_file, dev_file, test_file
        )
        
        if train_dataset is None:
            self.logger.error("No training data available!")
            return None, None, {"eval_accuracy": 0.0, "eval_f1": 0.0}
        
        self.logger.info(f"Train: {len(train_dataset)} examples")
        if dev_dataset:
            self.logger.info(f"Dev: {len(dev_dataset)} examples")
        if test_dataset:
            self.logger.info(f"Test: {len(test_dataset)} examples")
        
        # Load model again (we already loaded it in prepare_dataset_from_files)
        # But we need the model instance for training
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(Path(output_dir) / "checkpoints"),
            eval_strategy="epoch" if dev_dataset else "no",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            logging_dir=str(Path(output_dir) / "logs"),
            logging_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True if dev_dataset else False,
            metric_for_best_model="accuracy",
            greater_is_better=True,
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if dev_dataset else None,
        )
        
        # Train
        self.logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Evaluate on test
        test_results = {"eval_accuracy": 0.0, "eval_f1": 0.0}
        if test_dataset:
            test_results = trainer.evaluate(test_dataset)
            self.logger.info(f"Test results: {test_results}")
        
        # Save final model
        final_path = Path(output_dir) / "final_model"
        model.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        
        self.logger.info(f"Fine-tuning complete! Model saved to {final_path}")
        
        return model, tokenizer, test_results


if __name__ == "__main__":
    # Fine-tune using the saved CONLL-U files
    finetuner = GeEzPOSFineTuner(pretrained_model_path="bert-base-multilingual-cased")
    
    model, tokenizer, results = finetuner.finetune(
        train_file="./combined_pos_data/train.conllu",
        dev_file="./combined_pos_data/dev.conllu",
        test_file="./combined_pos_data/test.conllu",
        output_dir="./finetuned_pos",
        num_epochs=3
    )
    
    print(f"\n✅ Final accuracy: {results.get('eval_accuracy', 0):.4f}")
    print(f"✅ Final F1: {results.get('eval_f1', 0):.4f}")