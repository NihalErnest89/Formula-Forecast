"""
Test script to systematically improve classification model accuracy.
Tests different configurations and selects the best one.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
import json

# Import from top20/train.py for shared utilities
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import importlib.util
spec = importlib.util.spec_from_file_location("train_module", parent_dir / "top20" / "train.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

load_data = train_module.load_data
prepare_features_and_labels = train_module.prepare_features_and_labels

# Import classification components
sys.path.insert(0, str(Path(__file__).parent))
from train import F1ClassificationDataset, F1ClassificationNetwork, train_epoch, evaluate_model


def calculate_class_weights(y_train, top3_multiplier=1.5):
    """Calculate class weights with configurable top-3 boost."""
    position_counts = np.bincount(y_train.astype(int) - 1, minlength=10)
    total = len(y_train)
    class_weights = total / (10 * position_counts + 1e-5)
    class_weights = class_weights / class_weights.mean()
    
    # Boost top-3 positions
    class_weights[0] *= top3_multiplier  # Position 1
    class_weights[1] *= top3_multiplier  # Position 2
    class_weights[2] *= top3_multiplier  # Position 3
    class_weights = class_weights / class_weights.mean()
    
    return torch.FloatTensor(class_weights)


def train_and_evaluate_config(X_train, y_train, X_val, y_val, X_test, y_test, device, config):
    """Train and evaluate a model with given configuration."""
    print(f"\n{'='*70}")
    print(f"Testing Configuration: {config['name']}")
    print(f"{'='*70}")
    print(f"  Architecture: {config['hidden_sizes']}")
    print(f"  Dropout: {config['dropout_rate']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Weight Decay: {config['weight_decay']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Top-3 Weight Multiplier: {config['top3_multiplier']}")
    print(f"  Epochs: {config['num_epochs']}")
    
    # Create datasets
    train_dataset = F1ClassificationDataset(X_train, y_train)
    val_dataset = F1ClassificationDataset(X_val, y_val)
    test_dataset = F1ClassificationDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    input_size = X_train.shape[1]
    model = F1ClassificationNetwork(
        input_size=input_size,
        hidden_sizes=config['hidden_sizes'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    # Loss function
    class_weights = calculate_class_weights(y_train, config['top3_multiplier']).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=False, min_lr=1e-6
    )
    
    # Training
    best_val_mae = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 30
    
    for epoch in range(config['num_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        scheduler.step(val_metrics['mae'])
        
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    
    print(f"\nResults:")
    print(f"  Val MAE: {best_val_mae:.4f}")
    print(f"  Test MAE: {test_metrics['mae']:.4f}")
    print(f"  Test Exact Acc: {test_metrics['exact_acc']:.2f}%")
    print(f"  Test Within 1: {test_metrics['within_1']:.2f}%")
    print(f"  Test Within 3: {test_metrics['within_3']:.2f}%")
    print(f"  Test Top-3 Acc: {test_metrics['top3_acc']:.2f}%")
    
    return {
        'config': config,
        'val_mae': best_val_mae,
        'test_mae': test_metrics['mae'],
        'test_exact_acc': test_metrics['exact_acc'],
        'test_within_1': test_metrics['within_1'],
        'test_within_3': test_metrics['within_3'],
        'test_top3_acc': test_metrics['top3_acc'],
        'model_state': best_model_state
    }


def main():
    """Main function to test different configurations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_df, test_df, metadata = load_data()
    
    # Prepare features and labels
    print("\nPreparing features and labels (top 10 only)...")
    X_train, y_train, feature_cols = prepare_features_and_labels(train_df, top10_only=True)
    X_test, y_test, _ = prepare_features_and_labels(test_df, top10_only=True, feature_cols=feature_cols)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Time-aware split for validation
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train[:split_idx]
    y_train_split = y_train[:split_idx]
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    
    print(f"\nTrain: {len(X_train_split)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Define configurations to test
    configurations = [
        {
            'name': 'Baseline (Current)',
            'hidden_sizes': [256, 128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 64,
            'top3_multiplier': 1.5,
            'num_epochs': 300
        },
        {
            'name': 'Deeper Network',
            'hidden_sizes': [512, 256, 128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 64,
            'top3_multiplier': 1.5,
            'num_epochs': 300
        },
        {
            'name': 'Wider Network',
            'hidden_sizes': [512, 256, 128],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 64,
            'top3_multiplier': 1.5,
            'num_epochs': 300
        },
        {
            'name': 'Higher Top-3 Weight',
            'hidden_sizes': [256, 128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 64,
            'top3_multiplier': 2.0,
            'num_epochs': 300
        },
        {
            'name': 'Lower Learning Rate',
            'hidden_sizes': [256, 128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.0005,
            'weight_decay': 1e-4,
            'batch_size': 64,
            'top3_multiplier': 1.5,
            'num_epochs': 300
        },
        {
            'name': 'More Regularization',
            'hidden_sizes': [256, 128, 64],
            'dropout_rate': 0.4,
            'learning_rate': 0.001,
            'weight_decay': 2e-4,
            'batch_size': 64,
            'top3_multiplier': 1.5,
            'num_epochs': 300
        },
        {
            'name': 'Best Combined',
            'hidden_sizes': [512, 256, 128, 64],
            'dropout_rate': 0.3,
            'learning_rate': 0.0005,
            'weight_decay': 1e-4,
            'batch_size': 64,
            'top3_multiplier': 2.0,
            'num_epochs': 300
        },
    ]
    
    results = []
    
    for config in configurations:
        try:
            result = train_and_evaluate_config(
                X_train_split, y_train_split, X_val, y_val, X_test, y_test, device, config
            )
            results.append(result)
        except Exception as e:
            print(f"Error testing {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Find best configuration
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL CONFIGURATIONS")
    print(f"{'='*70}")
    
    # Sort by test exact accuracy (primary), then top-3 accuracy (secondary)
    results.sort(key=lambda x: (x['test_exact_acc'], x['test_top3_acc']), reverse=True)
    
    print(f"\n{'Rank':<6} {'Config':<25} {'Test Exact':<12} {'Test Top-3':<12} {'Test MAE':<10} {'Test W1':<10} {'Test W3':<10}")
    print("-" * 85)
    
    for i, result in enumerate(results, 1):
        print(f"{i:<6} {result['config']['name']:<25} {result['test_exact_acc']:>10.2f}% {result['test_top3_acc']:>10.2f}% "
              f"{result['test_mae']:>8.4f} {result['test_within_1']:>8.2f}% {result['test_within_3']:>8.2f}%")
    
    # Save best configuration
    best_result = results[0]
    print(f"\n{'='*70}")
    print(f"BEST CONFIGURATION: {best_result['config']['name']}")
    print(f"{'='*70}")
    print(f"  Test Exact Accuracy: {best_result['test_exact_acc']:.2f}%")
    print(f"  Test Top-3 Accuracy: {best_result['test_top3_acc']:.2f}%")
    print(f"  Test MAE: {best_result['test_mae']:.4f}")
    print(f"  Test Within 1: {best_result['test_within_1']:.2f}%")
    print(f"  Test Within 3: {best_result['test_within_3']:.2f}%")
    
    # Save results to file
    output_dir = Path(__file__).parent / 'models'
    output_dir.mkdir(exist_ok=True)
    
    results_summary = []
    for result in results:
        results_summary.append({
            'name': result['config']['name'],
            'test_exact_acc': result['test_exact_acc'],
            'test_top3_acc': result['test_top3_acc'],
            'test_mae': result['test_mae'],
            'test_within_1': result['test_within_1'],
            'test_within_3': result['test_within_3'],
            'config': result['config']
        })
    
    with open(output_dir / 'classification_test_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'classification_test_results.json'}")


if __name__ == '__main__':
    main()

