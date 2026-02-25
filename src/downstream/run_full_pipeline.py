# run_full_pipeline.py

from src.downstream.unified_pos_data import UnifiedPOSDataLoader
from src.downstream.pretrain_bert import Ge ezBertPreTrainer
from src.downstream.finetune_pos import Ge ezPOSFineTuner
from src.logger import get_logger

logger = get_logger()

def run_full_pipeline():
    """Run complete pre-training + fine-tuning pipeline"""
    
    # Step 1: Create combined datasets
    logger.info("=" * 60)
    logger.info("STEP 1: Creating combined datasets")
    logger.info("=" * 60)
    
    loader = UnifiedPOSDataLoader()
    pretrain_file = loader.create_pretraining_corpus("./pretrain_corpus.txt")
    train, dev, test = loader.create_pos_dataset("./combined_pos_data")
    
    # Step 2: Pre-train BERT
    logger.info("=" * 60)
    logger.info("STEP 2: Pre-training BERT on Ge'ez corpus")
    logger.info("=" * 60)
    
    pretrainer = Ge ezBertPreTrainer(output_dir="./geez_bert_pretrained")
    model, tokenizer = pretrainer.pretrain(
        corpus_file=pretrain_file,
        vocab_size=32000,
        num_epochs=5  # Increase for real training
    )
    
    # Step 3: Fine-tune on POS
    logger.info("=" * 60)
    logger.info("STEP 3: Fine-tuning on POS task")
    logger.info("=" * 60)
    
    finetuner = Ge ezPOSFineTuner(
        pretrained_model_path="./geez_bert_pretrained/final_model"
    )
    
    model, tokenizer, results = finetuner.finetune(
        output_dir="./finetuned_pos",
        num_epochs=10  # More epochs for fine-tuning
    )
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info(f"Final POS Accuracy: {results['eval_accuracy']:.4f}")
    logger.info(f"Final POS F1 Score: {results['eval_f1']:.4f}")
    logger.info("=" * 60)
    
    return results

if __name__ == "__main__":
    run_full_pipeline()