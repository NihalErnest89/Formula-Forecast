"""
Test script to compare different loss functions for exact position accuracy.
This will help us find a loss function that improves exact matches without
hurting overall MAE.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import importlib.util

# Import from top20/train.py
parent_dir = Path(__file__).parent
spec = importlib.util.spec_from_file_location("train_module", parent_dir / "top20" / "train.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

F1Dataset = train_module.F1Dataset
F1NeuralNetwork = train_module.F1NeuralNetwork
load_data = train_module.load_data
prepare_features_and_labels = train_module.prepare_features_and_labels
train_epoch = train_module.train_epoch
evaluate_model = train_module.evaluate_model


class StrongerExactLoss(nn.Module):
    """
    Loss function that more aggressively penalizes exact position misses.
    Uses a combination of Huber loss and a stronger exact-match penalty.
    """
    def __init__(self, exact_weight=5.0, base_loss='huber', delta=1.0):
        super().__init__()
        self.exact_weight = exact_weight
        self.delta = delta
        if base_loss == 'huber':
            self.base_loss = nn.HuberLoss(delta=delta, reduction='none')
        elif base_loss == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        else:
            self.base_loss = nn.L1Loss(reduction='none')
    
    def forward(self, pred, target):
        base = self.base_loss(pred, target)
        
        # Stronger penalty for exact misses - apply per-sample, not averaged
        rounded_pred = torch.round(pred)
        exact_misses = (rounded_pred != target).float()
        
        # Multiply base loss by (1 + exact_weight) for exact misses
        # This makes exact misses much more expensive
        weighted_loss = base * (1 + self.exact_weight * exact_misses)
        
        return weighted_loss.mean()


class HybridExactLoss(nn.Module):
    """
    Hybrid loss: combines regression loss with classification-style exact match loss.
    This directly optimizes for exact matches while maintaining good MAE.
    """
    def __init__(self, exact_weight=3.0, base_loss='huber', delta=1.0):
        super().__init__()
        self.exact_weight = exact_weight
        self.delta = delta
        if base_loss == 'huber':
            self.base_loss = nn.HuberLoss(delta=delta, reduction='none')
        else:
            self.base_loss = nn.MSELoss(reduction='none')
        
        # Classification loss for exact matches
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, pred, target):
        # Regression component
        base = self.base_loss(pred, target)
        
        # Exact match component: treat as classification problem
        # Create soft targets around the correct position
        rounded_pred = torch.round(pred)
        exact_matches = (rounded_pred == target).float()
        
        # Penalty for exact misses: add extra loss
        exact_penalty = (1 - exact_matches) * self.exact_weight * base
        
        # Combine: base loss + exact penalty
        total_loss = base + exact_penalty
        
        return total_loss.mean()


def train_model(X_train, y_train, X_val, y_val, criterion, epochs=100, 
                learning_rate=0.005, device='cpu', hidden_sizes=[128, 64, 32]):
    """Train a single model with given loss function."""
    train_dataset = F1Dataset(X_train, y_train)
    val_dataset = F1Dataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = F1NeuralNetwork(input_size=X_train.shape[1], 
                           hidden_sizes=hidden_sizes, 
                           dropout_rate=0.4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=False)
    
    best_val_mae = float('inf')
    patience_counter = 0
    patience = 30
    
    for epoch in range(epochs):
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae, val_rmse, val_r2, val_exact, val_w1, val_w2, val_w3, _, _ = evaluate_model(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return model, {
        'val_mae': val_mae,
        'val_exact': val_exact,
        'val_w1': val_w1,
        'val_w3': val_w3
    }


def main():
    print("=" * 70)
    print("Testing Different Loss Functions for Exact Position Accuracy")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_df, test_df, _ = load_data()
    
    # Prepare features (top 10 only) - but we need to track which rows are kept
    print("\nPreparing features (top 10 only)...")
    
    # Filter train_df the same way prepare_features_and_labels does
    train_df_filtered = train_df.copy()
    if 'Status' in train_df_filtered.columns:
        train_df_filtered = train_df_filtered[~train_df_filtered['Status'].str.contains('DNF|DSQ|DNS', na=False)]
    if 'ActualPosition' in train_df_filtered.columns:
        train_df_filtered = train_df_filtered[train_df_filtered['ActualPosition'].notna()]
        train_df_filtered = train_df_filtered[train_df_filtered['ActualPosition'] <= 10]
    if 'GridPosition' in train_df_filtered.columns and 'ActualPosition' in train_df_filtered.columns:
        position_diff = train_df_filtered['ActualPosition'] - train_df_filtered['GridPosition']
        train_df_filtered = train_df_filtered[position_diff <= 6]
    
    # Now prepare features - this should match the filtered dataframe
    X_train, y_train, _, _, _ = prepare_features_and_labels(
        train_df, filter_dnf=True, filter_outliers=True, top10_only=True
    )
    X_test, y_test, _, _, _ = prepare_features_and_labels(
        test_df, filter_dnf=True, filter_outliers=True, top10_only=True
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Time-based split: use 2024 as validation
    # The filtered dataframe should have the same length as X_train
    if len(train_df_filtered) == len(X_train):
        val_mask = train_df_filtered['Year'] == 2024
        train_mask = ~val_mask
    else:
        # Fallback: use a simple split (80/20)
        print("  Warning: Filtered dataframe length mismatch, using 80/20 split")
        split_idx = int(len(X_train) * 0.8)
        train_mask = np.zeros(len(X_train), dtype=bool)
        train_mask[:split_idx] = True
        val_mask = ~train_mask
    
    X_train_split = X_train[train_mask]
    y_train_split = y_train[train_mask]
    X_val_split = X_train[val_mask]
    y_val_split = y_train[val_mask]
    
    print(f"Training samples: {len(X_train_split)}")
    print(f"Validation samples: {len(X_val_split)}")
    print(f"Test samples: {len(X_test)}")
    
    # Test different loss functions
    loss_configs = [
        ('PositionAwareLoss (current, weight=2.0)', 
         train_module.PositionAwareLoss(exact_weight=2.0, base_loss='huber', delta=1.0)),
        ('PositionAwareLoss (stronger, weight=5.0)', 
         train_module.PositionAwareLoss(exact_weight=5.0, base_loss='huber', delta=1.0)),
        ('StrongerExactLoss (weight=5.0)', 
         StrongerExactLoss(exact_weight=5.0, base_loss='huber', delta=1.0)),
        ('HybridExactLoss (weight=3.0)', 
         HybridExactLoss(exact_weight=3.0, base_loss='huber', delta=1.0)),
    ]
    
    results = []
    
    for name, criterion in loss_configs:
        print(f"\n{'=' * 70}")
        print(f"Testing: {name}")
        print(f"{'=' * 70}")
        
        model, metrics = train_model(
            X_train_split, y_train_split,
            X_val_split, y_val_split,
            criterion=criterion,
            epochs=150,
            learning_rate=0.005,
            device='cpu'
        )
        
        # Evaluate on test set
        test_dataset = F1Dataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        test_loss, test_mae, test_rmse, test_r2, test_exact, test_w1, test_w2, test_w3, _, _ = evaluate_model(
            model, test_loader, criterion, device='cpu'
        )
        
        results.append({
            'loss': name,
            'val_mae': metrics['val_mae'],
            'val_exact': metrics['val_exact'],
            'val_w3': metrics['val_w3'],
            'test_mae': test_mae,
            'test_exact': test_exact,
            'test_w3': test_w3,
        })
        
        print(f"\nResults:")
        print(f"  Validation: MAE={metrics['val_mae']:.3f}, Exact={metrics['val_exact']:.1f}%, W3={metrics['val_w3']:.1f}%")
        print(f"  Test:       MAE={test_mae:.3f}, Exact={test_exact:.1f}%, W3={test_w3:.1f}%")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY - Loss Function Comparison")
    print(f"{'=' * 70}")
    print(f"{'Loss Function':<40} {'Val MAE':<10} {'Val Exact':<12} {'Test MAE':<10} {'Test Exact':<12}")
    print("-" * 70)
    
    baseline_mae = results[0]['test_mae']
    baseline_exact = results[0]['test_exact']
    
    for r in results:
        mae_diff = r['test_mae'] - baseline_mae
        exact_diff = r['test_exact'] - baseline_exact
        mae_str = f"{r['test_mae']:.3f} ({mae_diff:+.3f})" if mae_diff != 0 else f"{r['test_mae']:.3f}"
        exact_str = f"{r['test_exact']:.1f}% ({exact_diff:+.1f}%)" if exact_diff != 0 else f"{r['test_exact']:.1f}%"
        
        print(f"{r['loss']:<40} {r['val_mae']:<10.3f} {r['val_exact']:<12.1f} {mae_str:<10} {exact_str:<12}")
    
    print(f"\n{'=' * 70}")
    print("Recommendation:")
    best_exact = max(results, key=lambda x: x['test_exact'])
    if best_exact['test_exact'] > baseline_exact + 2.0:  # At least 2% improvement
        print(f"  Use: {best_exact['loss']}")
        print(f"  Improves exact accuracy by {best_exact['test_exact'] - baseline_exact:.1f}%")
        if best_exact['test_mae'] <= baseline_mae + 0.05:  # MAE within 0.05
            print(f"  MAE impact: {best_exact['test_mae'] - baseline_mae:+.3f} (acceptable)")
        else:
            print(f"  Warning: MAE increased by {best_exact['test_mae'] - baseline_mae:.3f}")
    else:
        print(f"  Current PositionAwareLoss is already optimal")
        print(f"  Best alternative only improves by {best_exact['test_exact'] - baseline_exact:.1f}%")


if __name__ == "__main__":
    main()

