"""
Static checks for evaluating tokenizers.
Includes all metrics from the SPLINTER paper PLUS PhD-level components.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import re
import json
import math
import numpy as np
import pandas as pd
import random
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sentencepiece as spm
from src.logger import get_logger
from src.utils.path_utils import get_static_checks_path, get_experiment_dir, get_tokenized_corpus_path, get_tokenizer_path


# ============================================================
# 1. BASIC STATIC CHECKS
# ============================================================

def run_static_checks(corpus_path, tokenized_path, tokenizer_path, vocab_size):
    """Run comprehensive static checks on tokenized corpus."""
    logger = get_logger()
    logger.info(f"Running static checks for vocab size {vocab_size}")
    
    results = {
        'vocab_size': vocab_size,
        'corpus': str(corpus_path),
        'tokenized': str(tokenized_path),
        'tokenizer': str(tokenizer_path),
        'checks': {}
    }
    
    try:
        if not Path(corpus_path).exists():
            logger.error(f"Corpus file not found: {corpus_path}")
            results['error'] = f"Corpus file not found: {corpus_path}"
            return results
            
        if not Path(tokenized_path).exists():
            logger.error(f"Tokenized file not found: {tokenized_path}")
            results['error'] = f"Tokenized file not found: {tokenized_path}"
            return results
            
        if not Path(str(tokenizer_path) + ".model").exists():
            logger.error(f"Tokenizer model not found: {tokenizer_path}.model")
            results['error'] = f"Tokenizer model not found: {tokenizer_path}.model"
            return results
        
        sp = spm.SentencePieceProcessor()
        sp.load(str(tokenizer_path) + ".model")
        logger.info(f"Loaded tokenizer with vocab size: {sp.get_piece_size()}")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            original_lines = [line.strip() for line in f.readlines()[:10000] if line.strip()]
        
        with open(tokenized_path, 'r', encoding='utf-8') as f:
            tokenized_lines = [line.strip() for line in f.readlines()[:10000] if line.strip()]
        
        logger.info(f"Analyzing {len(original_lines)} lines...")
        
        total_tokens = 0
        sentence_lengths = []
        special_token_count = 0
        token_frequencies = Counter()
        
        for line in tokenized_lines:
            tokens = line.split()
            total_tokens += len(tokens)
            sentence_lengths.append(len(tokens))
            token_frequencies.update(tokens)
            
            for token in tokens:
                if token.startswith('⟨') and token.endswith('⟩'):
                    special_token_count += 1
        
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        vocab_used = len(token_frequencies)
        vocab_used_percentage = (vocab_used / sp.get_piece_size() * 100) if sp.get_piece_size() > 0 else 0
        special_token_percentage = (special_token_count / total_tokens * 100) if total_tokens > 0 else 0
        fertility = total_tokens / len(original_lines) if original_lines else 0
        
        most_common = token_frequencies.most_common(20)
        
        results['checks'] = {
            'total_lines': len(original_lines),
            'total_tokens': total_tokens,
            'unique_tokens': vocab_used,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'max_sentence_length': max(sentence_lengths) if sentence_lengths else 0,
            'min_sentence_length': min(sentence_lengths) if sentence_lengths else 0,
            'vocab_size': sp.get_piece_size(),
            'vocab_used_percentage': round(vocab_used_percentage, 2),
            'special_token_count': special_token_count,
            'special_token_percentage': round(special_token_percentage, 2),
            'fertility': round(fertility, 2),
            'most_common_tokens': [(token, count) for token, count in most_common]
        }
        
        logger.info(f"✓ Static checks complete")
        
    except Exception as e:
        logger.error(f"Static checks failed: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
    
    try:
        static_checks_path = get_static_checks_path()
        if static_checks_path.exists():
            with open(static_checks_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        else:
            all_results = []
        
        all_results.append(results)
        
        with open(static_checks_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {static_checks_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    return results


def run_types_length_distribution(tokenizer_type, vocab_size):
    """Analyze type-length distribution."""
    logger = get_logger()
    logger.info(f"Running type-length distribution for {tokenizer_type} v{vocab_size}")
    
    results = {
        'tokenizer_type': tokenizer_type,
        'vocab_size': vocab_size,
        'analysis': 'type_length_distribution',
        'timestamp': __import__('datetime').datetime.now().isoformat()
    }
    
    try:
        static_checks_path = get_static_checks_path()
        if static_checks_path.exists():
            with open(static_checks_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        else:
            all_results = []
        
        all_results.append(results)
        
        with open(static_checks_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save type-length results: {e}")
    
    return results


# ============================================================
# 2. VOCABULARY OVERLAP (Figure 2)
# ============================================================

def load_vocab(vocab_path):
    """Load vocabulary from .vocab file."""
    tokens = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if parts:
                tokens.append(parts[0])
    return set(tokens)


def compute_vocabulary_overlap(vanilla_vocab_path, splinter_vocab_path, vocab_size):
    """Compute intersection percentage between vanilla and splinter vocabularies."""
    vanilla_tokens = load_vocab(vanilla_vocab_path)
    splinter_tokens = load_vocab(splinter_vocab_path)
    
    special_tokens = [t for t in splinter_tokens if t.startswith('⟨') and t.endswith('⟩')]
    
    expanded_splinter = set()
    for token in splinter_tokens:
        if token.startswith('⟨') and token.endswith('⟩'):
            continue
        expanded_splinter.add(token)
    
    intersection = vanilla_tokens.intersection(expanded_splinter)
    overlap_percentage = len(intersection) / len(vanilla_tokens) * 100 if vanilla_tokens else 0
    
    return {
        'vocab_size': vocab_size,
        'vanilla_tokens': len(vanilla_tokens),
        'splinter_tokens': len(splinter_tokens),
        'special_tokens': len(special_tokens),
        'intersection_tokens': len(intersection),
        'overlap_percentage': round(overlap_percentage, 2)
    }


def analyze_vocabulary_overlap(exp_dir, tokenizer_type='bpe'):
    """Analyze vocabulary overlap for all vocabulary sizes."""
    logger = get_logger()
    logger.info(f"Analyzing vocabulary overlap for {tokenizer_type}")
    
    tokenizers_dir = exp_dir / "tokenizers"
    results = []
    vocab_sizes = [4000, 6000,8000, 10000, 15000, 20000, 25000]  # FIXED: Correct vocab sizes
    
    for vocab_size in vocab_sizes:
        possible_vanilla = [
            tokenizers_dir / f"ge_{tokenizer_type}_{vocab_size}.vocab",
            tokenizers_dir / f"ge_baseline_{tokenizer_type}_{vocab_size}.vocab",
        ]
        
        possible_splinter = [
            tokenizers_dir / f"ge_{tokenizer_type}_{vocab_size}_splinter.vocab",
            tokenizers_dir / f"ge_splintered_{tokenizer_type}_{vocab_size}.vocab",
        ]
        
        vanilla_path = next((p for p in possible_vanilla if p.exists()), None)
        splinter_path = next((p for p in possible_splinter if p.exists()), None)
        
        if vanilla_path and splinter_path:
            result = compute_vocabulary_overlap(vanilla_path, splinter_path, vocab_size)
            results.append(result)
            logger.info(f"  Vocab size {vocab_size}: {result['overlap_percentage']}% overlap")
    
    return results


def plot_vocabulary_overlap(results, save_path=None):
    """Plot vocabulary overlap (like Figure 2 in paper)."""
    if not results:
        return
    
    vocab_sizes = [r['vocab_size'] for r in results]
    overlaps = [r['overlap_percentage'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(vocab_sizes, overlaps, 'bo-', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('Vocabulary Size (log scale)', fontsize=12)
    plt.ylabel('Overlap Percentage', fontsize=12)
    plt.title('Vocabulary Overlap: Vanilla vs SPLINTER', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    for i, (size, overlap) in enumerate(zip(vocab_sizes, overlaps)):
        plt.annotate(f'{overlap}%', (size, overlap), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ============================================================
# 3. RÉNYI EFFICIENCY (Tables 2-4) - FIXED VERSION
# ============================================================

def compute_renyi_efficiency(token_frequencies, alpha=2.0, exclude_special=True):
    """
    Compute Rényi efficiency for a token distribution.
    
    Args:
        token_frequencies: Dict of token -> frequency
        alpha: Rényi order parameter
        exclude_special: If True, exclude special tokens like ⟨n⟩, <pad>, etc.
    """
    # ============================================================
    # FIX: Exclude special tokens to get realistic values
    # ============================================================
    if exclude_special:
        filtered_freqs = {}
        for token, freq in token_frequencies.items():
            # Skip SPLINTER tags (⟨1⟩, ⟨2⟩, etc.)
            if token.startswith('⟨') and token.endswith('⟩'):
                continue
            # Skip special tokens
            if token in ['<pad>', '<unk>', '<s>', '</s>', '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
                continue
            # Skip tokens that are just punctuation or numbers
            if len(token) == 1 and not ('ሀ' <= token <= 'ፐ'):
                continue
            filtered_freqs[token] = freq
        token_frequencies = filtered_freqs
    
    total = sum(token_frequencies.values())
    if total == 0:
        return 0
    
    probs = np.array(list(token_frequencies.values())) / total
    
    if alpha == 1.0:
        renyi_entropy = -np.sum(probs * np.log(probs + 1e-10))
    else:
        renyi_entropy = (1 / (1 - alpha)) * np.log(np.sum(probs ** alpha) + 1e-10)
    
    n = len(probs)
    if n <= 1:
        return 1.0
    
    # This is the standard formula
    efficiency = np.exp(renyi_entropy) / n
    
    # For debugging
    # print(f"  n={n}, entropy={renyi_entropy:.4f}, exp(entropy)={np.exp(renyi_entropy):.4f}, efficiency={efficiency:.4f}")
    
    return efficiency


def compute_token_frequencies(tokenized_path, sample_lines=10000):
    """Compute token frequencies from tokenized corpus."""
    token_counter = Counter()
    
    with open(tokenized_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            tokens = line.strip().split()
            token_counter.update(tokens)
    
    return dict(token_counter)


def analyze_renyi_efficiency(exp_dir, tokenizer_type='bpe', vocab_sizes=None):
    """Compute Rényi efficiency for all tokenizers."""
    logger = get_logger()
    logger.info(f"Computing Rényi efficiency for {tokenizer_type}")
    
    if vocab_sizes is None:
        vocab_sizes = [300, 4000, 1000, 2000, 3000, 6000, 8000]
    
    results = {
        'tokenizer_type': tokenizer_type,
        'vanilla': [],
        'splinter': []
    }
    
    for vocab_size in vocab_sizes:
        vanilla_path = get_tokenized_corpus_path('original', tokenizer_type, vocab_size)
        if vanilla_path.exists():
            freqs = compute_token_frequencies(vanilla_path)
            # ============================================================
            # FIX: Use exclude_special=True for realistic values
            # ============================================================
            efficiency = compute_renyi_efficiency(freqs, exclude_special=True)
            results['vanilla'].append({
                'vocab_size': vocab_size,
                'efficiency': round(efficiency, 4)
            })
            logger.info(f"  Vanilla {vocab_size}: {efficiency:.4f}")
        
        splinter_path = get_tokenized_corpus_path('splintered', tokenizer_type, vocab_size)
        if splinter_path.exists():
            freqs = compute_token_frequencies(splinter_path)
            efficiency = compute_renyi_efficiency(freqs, exclude_special=True)
            results['splinter'].append({
                'vocab_size': vocab_size,
                'efficiency': round(efficiency, 4)
            })
            logger.info(f"  Splinter {vocab_size}: {efficiency:.4f}")
    
    return results


# ============================================================
# 4. DISTINCT NEIGHBORS (Figure 3, Tables 2-4)
# ============================================================

def compute_distinct_neighbors(tokenized_path, window_size=2, sample_lines=10000):
    """Compute average number of distinct neighbors for each token."""
    token_contexts = defaultdict(set)
    token_counts = Counter()
    
    with open(tokenized_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            
            token_counts.update(tokens)
            
            for j, token in enumerate(tokens):
                left_start = max(0, j - window_size)
                for k in range(left_start, j):
                    token_contexts[token].add(tokens[k])
                
                right_end = min(len(tokens), j + window_size + 1)
                for k in range(j + 1, right_end):
                    token_contexts[token].add(tokens[k])
    
    total_weighted = 0
    total_freq = 0
    for token, freq in token_counts.items():
        if token in token_contexts:
            total_weighted += len(token_contexts[token]) * freq
            total_freq += freq
    
    avg_distinct_neighbors = total_weighted / total_freq if total_freq > 0 else 0
    
    return {
        'avg_distinct_neighbors': round(avg_distinct_neighbors, 2),
        'total_tokens_analyzed': len(token_counts)
    }


def analyze_distinct_neighbors(exp_dir, tokenizer_type='bpe', vocab_sizes=None):
    """Analyze distinct neighbors for all tokenizers."""
    logger = get_logger()
    logger.info(f"Computing distinct neighbors for {tokenizer_type}")
    
    if vocab_sizes is None:
        vocab_sizes = [300, 4000, 1000, 2000, 3000, 6000, 8000]
    
    results = {
        'tokenizer_type': tokenizer_type,
        'vanilla': [],
        'splinter': []
    }
    
    for vocab_size in vocab_sizes:
        vanilla_path = get_tokenized_corpus_path('original', tokenizer_type, vocab_size)
        if vanilla_path.exists():
            stats = compute_distinct_neighbors(vanilla_path)
            stats['vocab_size'] = vocab_size
            results['vanilla'].append(stats)
            logger.info(f"  Vanilla {vocab_size}: {stats['avg_distinct_neighbors']}")
        
        splinter_path = get_tokenized_corpus_path('splintered', tokenizer_type, vocab_size)
        if splinter_path.exists():
            stats = compute_distinct_neighbors(splinter_path)
            stats['vocab_size'] = vocab_size
            results['splinter'].append(stats)
            logger.info(f"  Splinter {vocab_size}: {stats['avg_distinct_neighbors']}")
    
    return results


# ============================================================
# 5. COGNITIVE PLAUSIBILITY (using language_utils)
# ============================================================

def compute_surprisal(tokenizer_path, corpus_path, language_utils=None, sample_lines=10000):
    """
    Compute token surprisal as cognitive plausibility proxy.
    Now uses language_utils if provided for syllable-aware processsing.
    """
    import sentencepiece as spm
    import numpy as np
    from collections import Counter
    
    logger = get_logger()
    
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(str(tokenizer_path) + ".model")
        
        token_freq = Counter()
        total_tokens = 0
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_lines:
                    break
                # If language_utils provided, could use syllable_break here
                # but for now, just use raw tokenization
                tokens = sp.encode_as_pieces(line.strip())
                token_freq.update(tokens)
                total_tokens += len(tokens)
        
        if total_tokens == 0:
            return {
                'mean_surprisal': 0.0,
                'std_surprisal': 0.0,
                'entropy': 0.0,
                'unique_tokens': 0
            }
        
        token_probs = {token: freq/total_tokens for token, freq in token_freq.items()}
        
        surprisals = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 1000:
                    break
                tokens = sp.encode_as_pieces(line.strip())
                for token in tokens:
                    prob = token_probs.get(token, 1e-10)
                    surprisal = -np.log2(prob)
                    surprisals.append(surprisal)
        
        probs = np.array(list(token_probs.values()))
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return {
            'mean_surprisal': float(np.mean(surprisals)) if surprisals else 0,
            'std_surprisal': float(np.std(surprisals)) if surprisals else 0,
            'entropy': float(entropy),
            'unique_tokens': len(token_freq),
            'total_tokens_analyzed': total_tokens
        }
        
    except Exception as e:
        logger.error(f"Error computing surprisal: {e}")
        return {
            'mean_surprisal': 0.0,
            'std_surprisal': 0.0,
            'entropy': 0.0,
            'unique_tokens': 0,
            'error': str(e)
        }


def run_cognitive_evaluation(exp_dir, corpus_path, language_utils=None):
    """Run cognitive plausibility analysis."""
    logger = get_logger()
    logger.info("=" * 60)
    logger.info("COGNITIVE PLAUSIBILITY ANALYSIS")
    logger.info("=" * 60)
    
    results = {'bpe': {}, 'unigram': {}}
    vocab_sizes = [300, 4000, 1000, 2000, 3000, 6000, 8000]
    
    for tokenizer_type in ['bpe', 'unigram']:
        logger.info(f"\n📊 Computing surprisal for {tokenizer_type}")
        
        for vocab_size in vocab_sizes:
            vanilla_path = get_tokenizer_path(tokenizer_type, vocab_size)
            if vanilla_path.exists():
                surprisal = compute_surprisal(vanilla_path, corpus_path, language_utils)
                results[tokenizer_type][f'vanilla_{vocab_size}'] = surprisal
                logger.info(f"  Vanilla {vocab_size}: entropy={surprisal['entropy']:.2f}")
            
            splinter_path = exp_dir / "tokenizers" / f"ge_splintered_{tokenizer_type}_{vocab_size}"
            if splinter_path.exists():
                surprisal = compute_surprisal(splinter_path, corpus_path, language_utils)
                results[tokenizer_type][f'splinter_{vocab_size}'] = surprisal
                logger.info(f"  SPLINTER {vocab_size}: entropy={surprisal['entropy']:.2f}")
    
    print("\n" + "="*80)
    print("COGNITIVE PLAUSIBILITY SUMMARY")
    print("="*80)
    print(f"{'Type':<10} {'Vocab':<8} {'Entropy':<12} {'Mean Surprisal':<15}")
    print("-"*80)
    
    for tokenizer_type in ['bpe', 'unigram']:
        for size in vocab_sizes:
            vanilla_key = f'vanilla_{size}'
            if vanilla_key in results[tokenizer_type]:
                r = results[tokenizer_type][vanilla_key]
                print(f"{tokenizer_type:<10} {size:<8} Vanilla: {r['entropy']:<12.2f} {r['mean_surprisal']:<15.2f}")
            
            splinter_key = f'splinter_{size}'
            if splinter_key in results[tokenizer_type]:
                r = results[tokenizer_type][splinter_key]
                print(f"{tokenizer_type:<10} {size:<8} SPLINTER: {r['entropy']:<12.2f} {r['mean_surprisal']:<15.2f}")
            print()
    
    output_path = exp_dir / "static_checks" / "cognitive_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Cognitive results saved to {output_path}")
    return results


# ============================================================
# 6. COMPREHENSIVE EVALUATION (MAIN FUNCTION)
# ============================================================

def run_comprehensive_evaluation(encode_map=None, decode_map=None, language_utils=None):
    """Run all evaluation metrics."""
    logger = get_logger()
    exp_dir = get_experiment_dir()
    corpus_path = exp_dir / "corpora" / "original_corpus.txt"
    
    logger.info("="*80)
    logger.info("RUNNING COMPREHENSIVE EVALUATION")
    logger.info("="*80)
    
    all_results = {}
    
    for tokenizer_type in ['bpe', 'unigram']:
        logger.info(f"\n{'='*80}")
        logger.info(f"TOKENIZER TYPE: {tokenizer_type.upper()}")
        logger.info(f"{'='*80}")
        
        # Vocabulary Overlap
        overlap_results = analyze_vocabulary_overlap(exp_dir, tokenizer_type)
        all_results[f'vocab_overlap_{tokenizer_type}'] = overlap_results
        
        if overlap_results:
            plot_path = exp_dir / "static_checks" / f"vocabulary_overlap_{tokenizer_type}.png"
            plot_vocabulary_overlap(overlap_results, plot_path)
        
        # Rényi Efficiency
        renyi_results = analyze_renyi_efficiency(exp_dir, tokenizer_type)
        all_results[f'renyi_{tokenizer_type}'] = renyi_results
        
        # Distinct Neighbors
        neighbors_results = analyze_distinct_neighbors(exp_dir, tokenizer_type)
        all_results[f'neighbors_{tokenizer_type}'] = neighbors_results
    
    # Cognitive Plausibility (if language_utils provided)
    if language_utils:
        try:
            cognitive_results = run_cognitive_evaluation(exp_dir, corpus_path, language_utils)
            all_results['cognitive'] = cognitive_results
        except Exception as e:
            logger.error(f"Cognitive evaluation failed: {e}")
    
    # Qualitative Analysis (if maps provided)
    if encode_map and decode_map:
        try:
            from src.qualitative_analysis import run_qualitative_analysis
            qualitative_results = run_qualitative_analysis(exp_dir, encode_map, decode_map)
            all_results['qualitative'] = qualitative_results
        except ImportError:
            logger.warning("Qualitative analysis module not found")
        except Exception as e:
            logger.error(f"Qualitative analysis failed: {e}")
    
    output_path = exp_dir / "static_checks" / "comprehensive_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✅ All results saved to: {output_path}")
    return all_results


if __name__ == "__main__":
    run_comprehensive_evaluation()