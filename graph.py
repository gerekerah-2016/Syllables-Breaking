import json
import matplotlib.pyplot as plt
import numpy as np

# Your actual data from comprehensive_results.json
data = {
    "bpe": {
        "vanilla": [
            {"vocab_size": 800, "neighbors": 89.63},
            {"vocab_size": 1000, "neighbors": 88.25},
            {"vocab_size": 2000, "neighbors": 83.96},
            {"vocab_size": 5000, "neighbors": 82.72},
            {"vocab_size": 10000, "neighbors": 80.54},
            {"vocab_size": 15000, "neighbors": 72.40},
            {"vocab_size": 18000, "neighbors": 72.06}
        ],
        "splinter": [
            {"vocab_size": 800, "neighbors": 139.47},
            {"vocab_size": 1000, "neighbors": 156.06},
            {"vocab_size": 2000, "neighbors": 214.28},
            {"vocab_size": 5000, "neighbors": 292.18},
            {"vocab_size": 10000, "neighbors": 346.30},
            {"vocab_size": 15000, "neighbors": 365.56},
            {"vocab_size": 18000, "neighbors": 371.74}
        ]
    },
    "unigram": {
        "vanilla": [
            {"vocab_size": 800, "neighbors": 124.96},
            {"vocab_size": 1000, "neighbors": 129.55},
            {"vocab_size": 2000, "neighbors": 150.16},
            {"vocab_size": 5000, "neighbors": 199.51},
            {"vocab_size": 10000, "neighbors": 271.50},
            {"vocab_size": 15000, "neighbors": 307.18},
            {"vocab_size": 18000, "neighbors": 315.93}
        ],
        "splinter": [
            {"vocab_size": 800, "neighbors": 174.65},
            {"vocab_size": 1000, "neighbors": 195.61},
            {"vocab_size": 2000, "neighbors": 257.54},
            {"vocab_size": 5000, "neighbors": 304.36},
            {"vocab_size": 10000, "neighbors": 318.83},
            {"vocab_size": 15000, "neighbors": 316.72},
            {"vocab_size": 18000, "neighbors": 316.44}
        ]
    }
}

# Extract data for plotting
vocab_sizes = [800, 1000, 2000, 5000, 10000, 15000, 18000]

# BPE data
bpe_vanilla = [89.63, 88.25, 83.96, 82.72, 80.54, 72.40, 72.06]
bpe_splinter = [139.47, 156.06, 214.28, 292.18, 346.30, 365.56, 371.74]

# Unigram data
uni_vanilla = [124.96, 129.55, 150.16, 199.51, 271.50, 307.18, 315.93]
uni_splinter = [174.65, 195.61, 257.54, 304.36, 318.83, 316.72, 316.44]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ===== BPE Plot (left) =====
ax1.plot(vocab_sizes, bpe_vanilla, 'o-', linewidth=2.5, markersize=8, 
         label='Vanilla BPE', color='#1f77b4')
ax1.plot(vocab_sizes, bpe_splinter, 's-', linewidth=2.5, markersize=8, 
         label='SPLINTER BPE', color='#ff7f0e')

# Add value labels
for i, (v, s) in enumerate(zip(bpe_vanilla, bpe_splinter)):
    ax1.annotate(f'{v:.1f}', (vocab_sizes[i], v), 
                textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    ax1.annotate(f'{s:.1f}', (vocab_sizes[i], s), 
                textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='#ff7f0e')

# Highlight the 18000 vocab size result
ax1.axhline(y=371.74, xmin=0.8, xmax=1, linestyle='--', color='#ff7f0e', alpha=0.3)
ax1.axhline(y=72.06, xmin=0.8, xmax=1, linestyle='--', color='#1f77b4', alpha=0.3)

ax1.set_xlabel('Vocabulary Size', fontsize=12)
ax1.set_ylabel('Average Distinct Neighbors', fontsize=12)
ax1.set_title('BPE Tokenizer: Distinct Neighbors by Vocabulary Size\nGe\'ez Corpus', 
              fontsize=14, fontweight='bold')
ax1.set_xscale('log')
ax1.set_xticks(vocab_sizes)
ax1.set_xticklabels(vocab_sizes)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(0, 400)

# ===== Unigram Plot (right) =====
ax2.plot(vocab_sizes, uni_vanilla, 'o-', linewidth=2.5, markersize=8, 
         label='Vanilla Unigram', color='#1f77b4')
ax2.plot(vocab_sizes, uni_splinter, 's-', linewidth=2.5, markersize=8, 
         label='SPLINTER Unigram', color='#ff7f0e')

# Add value labels
for i, (v, s) in enumerate(zip(uni_vanilla, uni_splinter)):
    ax2.annotate(f'{v:.1f}', (vocab_sizes[i], v), 
                textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    ax2.annotate(f'{s:.1f}', (vocab_sizes[i], s), 
                textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='#ff7f0e')

ax2.set_xlabel('Vocabulary Size', fontsize=12)
ax2.set_ylabel('Average Distinct Neighbors', fontsize=12)
ax2.set_title('Unigram Tokenizer: Distinct Neighbors by Vocabulary Size\nGe\'ez Corpus', 
              fontsize=14, fontweight='bold')
ax2.set_xscale('log')
ax2.set_xticks(vocab_sizes)
ax2.set_xticklabels(vocab_sizes)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(0, 400)

plt.tight_layout()
plt.savefig('geez_distinct_neighbors_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print improvement statistics
print("=" * 60)
print("DISTINCT NEIGHBORS IMPROVEMENT - Ge'ez Corpus")
print("=" * 60)
print("\nBPE Tokenizer:")
print("-" * 40)
for i, size in enumerate(vocab_sizes):
    improvement = ((bpe_splinter[i] - bpe_vanilla[i]) / bpe_vanilla[i]) * 100
    print(f"Vocab {size:5d}: Vanilla {bpe_vanilla[i]:6.2f} → SPLINTER {bpe_splinter[i]:6.2f}  (+{improvement:5.1f}%)")

print("\nUnigram Tokenizer:")
print("-" * 40)
for i, size in enumerate(vocab_sizes):
    improvement = ((uni_splinter[i] - uni_vanilla[i]) / uni_vanilla[i]) * 100
    print(f"Vocab {size:5d}: Vanilla {uni_vanilla[i]:6.2f} → SPLINTER {uni_splinter[i]:6.2f}  (+{improvement:5.1f}%)")