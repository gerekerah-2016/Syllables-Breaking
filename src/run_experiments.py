#!/usr/bin/env python
"""
Run experiments for Ethiopic Syllable Breaking.
Author: Gebreslassie Teklu Reda
Date: 2026
"""

import os
import sys
import subprocess
from pathlib import Path


def setup_environment():
    """Setup environment and install dependencies."""
    print("Setting up environment...")
    
    # Install requirements
    requirements = [
        "sentencepiece",
        "datasets",
        "transformers",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn"
    ]
    
    for req in requirements:
        subprocess.run([sys.executable, "-m", "pip", "install", req])
    
    print("Environment setup complete!")


def create_data_directories():
    """Create data directories if they don't exist."""
    dirs = [
        "./Geez-Dataset",
        "./outputs",
        "./outputs/baseline_run",
        "./outputs/splintered_run",
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    print("Data directories created!")


def run_single_experiment(experiment_id: int):
    """
    Run a single experiment by ID.
    
    Args:
        experiment_id: 0 for baseline, 1 for splintered
    """
    print(f"\n{'='*60}")
    print(f"Running Experiment {experiment_id + 1}: {'SPLINTERED' if experiment_id == 1 else 'BASELINE'}")
    print(f"{'='*60}\n")
    
    # Set environment variable for SLURM (or use directly)
    os.environ['SLURM_ARRAY_TASK_ID'] = str(experiment_id + 1)
    
    # Run main.py
    result = subprocess.run([sys.executable, "main.py"])
    
    if result.returncode == 0:
        print(f"\n✓ Experiment {experiment_id + 1} completed successfully!")
    else:
        print(f"\n✗ Experiment {experiment_id + 1} failed with code {result.returncode}")
    
    return result.returncode


def run_all_experiments():
    """Run both baseline and splintered experiments."""
    print("\nStarting Ethiopic Syllable Breaking Pipeline")
    print("="*60)
    
    # Setup
    setup_environment()
    create_data_directories()
    
    # Run experiments
    baseline_result = run_single_experiment(0)
    if baseline_result != 0:
        print("Baseline experiment failed. Stopping.")
        return
    
    splintered_result = run_single_experiment(1)
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Baseline Experiment:   {'✓ SUCCESS' if baseline_result == 0 else '✗ FAILED'}")
    print(f"Splintered Experiment: {'✓ SUCCESS' if splintered_result == 0 else '✗ FAILED'}")
    
    # Show output locations
    print("\nOutput directories:")
    print(f"  Baseline:   ./outputs/baseline_run/")
    print(f"  Splintered: ./outputs/splintered_run/")
    print("\nCheck the following files for results:")
    print("  - Tokenizers:     outputs/*/tokenizers/")
    print("  - Corpora:        outputs/*/corpora/")
    print("  - Tokenized:      outputs/*/tokenized_corpora/")
    print("  - Static checks:  outputs/*/static_checks/results.json")
    print("  - Logs:           outputs/*/logs/pipeline.log")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific experiment
        experiment_id = int(sys.argv[1])
        if experiment_id in [0, 1]:
            run_single_experiment(experiment_id)
        else:
            print("Invalid experiment ID. Use 0 for baseline, 1 for splintered.")
    else:
        # Run both
        run_all_experiments()