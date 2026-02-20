"""
Master runner for all downstream tasks.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import json
from pathlib import Path
from src.logger import get_logger
from src.downstream.ner import train_ner_with_splinter
from src.downstream.pos import train_pos_with_splinter
from src.downstream.mt import train_mt_with_splinter
from src.downstream.classification import train_classification_with_splinter


def run_all_downstream_tasks(processor, exp_dir):
    """
    Run all downstream tasks and create comparison table.
    
    Args:
        processor: Your TextProcessorWithEncoding instance
        exp_dir: Experiment directory path
    """
    logger = get_logger()
    
    logger.info("\n" + "="*80)
    logger.info("RUNNING ALL DOWNSTREAM TASKS")
    logger.info("="*80)
    
    results = {}
    
    # Create output directories
    models_dir = Path(exp_dir) / "downstream_models"
    models_dir.mkdir(exist_ok=True)
    
    # 1. Named Entity Recognition
    logger.info("\n" + "="*60)
    logger.info("TASK 1: Named Entity Recognition (MasakhaNER)")
    logger.info("="*60)
    try:
        ner_results = train_ner_with_splinter(
            processor, 
            str(models_dir / "ner"),
            exp_dir
        )
        results['ner'] = ner_results
    except Exception as e:
        logger.error(f"NER failed: {e}")
        results['ner'] = {'error': str(e)}
    
    # 2. Part-of-Speech Tagging
    logger.info("\n" + "="*60)
    logger.info("TASK 2: POS Tagging (UD Amharic)")
    logger.info("="*60)
    try:
        pos_results = train_pos_with_splinter(
            processor,
            str(models_dir / "pos"),
            exp_dir
        )
        results['pos'] = pos_results
    except Exception as e:
        logger.error(f"POS failed: {e}")
        results['pos'] = {'error': str(e)}
    
    # 3. Machine Translation
    logger.info("\n" + "="*60)
    logger.info("TASK 3: Machine Translation (JW300)")
    logger.info("="*60)
    try:
        mt_results = train_mt_with_splinter(
            processor,
            str(models_dir / "mt"),
            exp_dir
        )
        results['mt'] = mt_results
    except Exception as e:
        logger.error(f"MT failed: {e}")
        results['mt'] = {'error': str(e)}
    
    # 4. Text Classification
    logger.info("\n" + "="*60)
    logger.info("TASK 4: Text Classification (Amharic News)")
    logger.info("="*60)
    try:
        cls_results = train_classification_with_splinter(
            processor,
            str(models_dir / "classification"),
            exp_dir
        )
        results['classification'] = cls_results
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        results['classification'] = {'error': str(e)}
    
    # Create comparison table like Table 5 in paper
    print("\n" + "="*80)
    print("DOWNSTREAM TASKS: Vanilla vs SPLINTER")
    print("="*80)
    print(f"{'Task':<20} {'Metric':<10} {'Vanilla':<12} {'SPLINTER':<12} {'Improvement':<12}")
    print("-"*80)
    
    for task, task_results in results.items():
        if 'error' in task_results:
            continue
            
        if task == 'ner':
            metric = 'F1'
            van = task_results['vanilla'].get('eval_f1', 0)
            spl = task_results['splinter'].get('eval_f1', 0)
        elif task == 'pos':
            metric = 'Acc'
            van = task_results['vanilla'].get('accuracy', 0)
            spl = task_results['splinter'].get('accuracy', 0)
        elif task == 'mt':
            metric = 'BLEU'
            van = task_results['vanilla'].get('bleu', 0)
            spl = task_results['splinter'].get('bleu', 0)
        elif task == 'classification':
            metric = 'Acc'
            van = task_results['vanilla'].get('accuracy', 0)
            spl = task_results['splinter'].get('accuracy', 0)
        else:
            continue
        
        if spl > 0 and van > 0:
            imp = ((spl - van) / van) * 100
            print(f"{task:<20} {metric:<10} {van:<12.3f} {spl:<12.3f} {imp:+.2f}%")
    
    # Save results
    output_path = Path(exp_dir) / "static_checks" / "downstream_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {output_path}")
    return results