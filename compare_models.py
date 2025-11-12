"""
Compare Neural Network and Random Forest models.
Loads results from both training runs and displays a comparison.
"""

import json
from pathlib import Path
import pandas as pd


def load_results():
    """Load results from both model types."""
    nn_results_path = Path('training_results.json')
    rf_results_path = Path('training_results_rf.json')
    
    nn_results = None
    rf_results = None
    
    if nn_results_path.exists():
        with open(nn_results_path, 'r', encoding='utf-8') as f:
            nn_results = json.load(f)
    
    if rf_results_path.exists():
        with open(rf_results_path, 'r', encoding='utf-8') as f:
            rf_results = json.load(f)
    
    return nn_results, rf_results


def compare_models():
    """Display comparison of both models."""
    print("=" * 70)
    print("Model Comparison: Neural Network vs Random Forest")
    print("=" * 70)
    
    nn_results, rf_results = load_results()
    
    if not nn_results and not rf_results:
        print("\nError: No training results found!")
        print("Please train at least one model:")
        print("  - Neural Network: python train.py")
        print("  - Random Forest: python train_rf.py")
        return
    
    print("\n" + "-" * 70)
    print("ACCURACY COMPARISON")
    print("-" * 70)
    
    if nn_results:
        print(f"\nNeural Network (Deep Learning):")
        print(f"  Validation Accuracy: {nn_results.get('validation_accuracy', 'N/A'):.4f}")
        print(f"  Test Accuracy: {nn_results.get('test_accuracy', 'N/A'):.4f if nn_results.get('test_accuracy') else 'N/A'}")
        print(f"  Model Type: Deep Neural Network (PyTorch)")
        print(f"  Architecture: 3 → 128 → 64 → 32 → {nn_results.get('num_classes', 'N')} classes")
        print(f"  Training: Multiple epochs with backpropagation")
    else:
        print("\nNeural Network: Not trained yet (run: python train.py)")
    
    if rf_results:
        print(f"\nRandom Forest (Traditional ML):")
        print(f"  Validation Accuracy: {rf_results.get('validation_accuracy', 'N/A'):.4f}")
        print(f"  Test Accuracy: {rf_results.get('test_accuracy', 'N/A'):.4f if rf_results.get('test_accuracy') else 'N/A'}")
        print(f"  Model Type: Random Forest (100 trees)")
        print(f"  Training: Single pass, no backpropagation")
    else:
        print("\nRandom Forest: Not trained yet (run: python train_rf.py)")
    
    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCE COMPARISON")
    print("-" * 70)
    
    if nn_results and nn_results.get('feature_importances'):
        print("\nNeural Network (First Layer Weights):")
        nn_importances = nn_results['feature_importances']
        for feature, importance in sorted(nn_importances.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {importance:.4f}")
    
    if rf_results and rf_results.get('feature_importances'):
        print("\nRandom Forest (Tree-based Importance):")
        rf_importances = rf_results['feature_importances']
        for feature, importance in sorted(rf_importances.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {importance:.4f}")
    
    print("\n" + "-" * 70)
    print("KEY DIFFERENCES")
    print("-" * 70)
    print("""
Neural Network (Deep Learning):
  ✓ Uses neurons, weights, and activations (ReLU)
  ✓ Trained with backpropagation and gradient descent
  ✓ Multiple epochs of training
  ✓ Learnable weight distribution
  ✓ Can capture complex non-linear patterns
  ✗ Requires more training time
  ✗ Less interpretable

Random Forest (Traditional ML):
  ✓ Ensemble of decision trees
  ✓ Fast training (single pass)
  ✓ Highly interpretable feature importance
  ✓ No backpropagation needed
  ✗ Fixed tree structure
  ✗ Less flexible than neural networks
    """)
    
    print("=" * 70)


if __name__ == "__main__":
    compare_models()

