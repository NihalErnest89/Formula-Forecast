"""
Compare Equal Weights vs He/Kaiming Initialization
Trains both models and compares their performance
"""

import sys
from pathlib import Path
import importlib.util

# Import from top20/train.py using the same method as top10/train.py
parent_dir = Path(__file__).parent
spec = importlib.util.spec_from_file_location("train_module", parent_dir / "top20" / "train.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

# Import the functions we need
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

# Use existing infrastructure
F1Dataset = train_module.F1Dataset
load_data = train_module.load_data
prepare_features_and_labels = train_module.prepare_features_and_labels
PositionAwareLoss = train_module.PositionAwareLoss
train_epoch = train_module.train_epoch
evaluate_model = train_module.evaluate_model

class F1Dataset(Dataset):
    """PyTorch Dataset for F1 prediction data."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Use the F1NeuralNetwork from top20/train.py which already supports equal_init parameter
F1NeuralNetwork = train_module.F1NeuralNetwork


def train_model(X_train, y_train, X_val, y_val, equal_init=False, epochs=100, device='cpu'):
    """Train a single model with specified initialization."""
    
    # Create datasets
    train_dataset = F1Dataset(X_train, y_train)
    val_dataset = F1Dataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = F1NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_sizes=[128, 64, 32],
        dropout_rate=0.4,
        equal_init=equal_init
    ).to(device)
    
    # Loss and optimizer
    criterion = PositionAwareLoss(exact_weight=5.0, base_loss='huber', delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    
    # Training loop
    best_val_mae = float('inf')
    patience = 30
    patience_counter = 0
    
    init_type = "Equal Weights" if equal_init else "He/Kaiming"
    print(f"\nTraining model with {init_type} initialization...")
    
    for epoch in range(epochs):
        # Training
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_mae, val_rmse, val_r2, val_exact, val_w1, val_w2, val_w3, _, _ = evaluate_model(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_mae)
        
        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train MAE={train_mae:.3f}, Val MAE={val_mae:.3f}, "
                  f"Val Exact={val_exact:.1%}, Val W3={val_w3:.1%}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Get final metrics
    model.eval()
    final_train_mae = train_mae  # Use last training MAE
    final_val_loss, final_val_mae, final_val_rmse, final_val_r2, final_val_exact, final_val_w1, final_val_w2, final_val_w3, _, _ = evaluate_model(
        model, val_loader, criterion, device
    )
    
    return model, {
        'best_val_mae': best_val_mae,
        'final_train_mae': final_train_mae,
        'final_val_mae': final_val_mae,
        'final_val_exact': final_val_exact,
        'final_val_w3': final_val_w3
    }


def evaluate_test_model(model, X_test, y_test, criterion, device='cpu'):
    """Evaluate model on test set."""
    test_dataset = F1Dataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    test_loss, test_mae, test_rmse, test_r2, test_exact, test_w1, test_w2, test_w3, _, _ = evaluate_model(
        model, test_loader, criterion, device
    )
    
    return {
        'mae': test_mae,
        'rmse': test_rmse,
        'r2': test_r2,
        'exact': test_exact,
        'within_1': test_w1,
        'within_2': test_w2,
        'within_3': test_w3
    }


def main():
    print("=" * 70)
    print("Weight Initialization Comparison: Equal Weights vs He/Kaiming")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    train_df, test_df, metadata = load_data()
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}\n")
    
    # Prepare features (top 10 only)
    print("Preparing features (top 10 positions only)...")
    X_train, y_train, label_encoder, _, feature_names = prepare_features_and_labels(
        train_df, filter_dnf=True, filter_outliers=True, top10_only=True
    )
    
    # Prepare test features
    X_test_raw, y_test, _, _, _ = prepare_features_and_labels(
        test_df, filter_dnf=True, filter_outliers=True, top10_only=True
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test_raw)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}\n")
    
    # Split training data for validation
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Train split: {len(X_train_split)}, Val split: {len(X_val_split)}\n")
    
    # Train Equal Weights model
    print("=" * 70)
    equal_model, equal_train_metrics = train_model(
        X_train_split, y_train_split, X_val_split, y_val_split,
        equal_init=True, epochs=150, device=device
    )
    
    # Train He/Kaiming model
    print("=" * 70)
    he_model, he_train_metrics = train_model(
        X_train_split, y_train_split, X_val_split, y_val_split,
        equal_init=False, epochs=150, device=device
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)
    
    criterion = PositionAwareLoss(exact_weight=5.0, base_loss='huber', delta=1.0)
    equal_test_metrics = evaluate_test_model(equal_model, X_test, y_test, criterion, device=device)
    he_test_metrics = evaluate_test_model(he_model, X_test, y_test, criterion, device=device)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print("\n--- Training Metrics ---")
    print(f"{'Metric':<25} {'Equal Weights':<20} {'He/Kaiming':<20} {'Difference':<15}")
    print("-" * 70)
    print(f"{'Val MAE (best)':<25} {equal_train_metrics['best_val_mae']:<20.4f} {he_train_metrics['best_val_mae']:<20.4f} {he_train_metrics['best_val_mae'] - equal_train_metrics['best_val_mae']:+.4f}")
    print(f"{'Val Exact Acc':<25} {equal_train_metrics['final_val_exact']:<20.2%} {he_train_metrics['final_val_exact']:<20.2%} {he_train_metrics['final_val_exact'] - equal_train_metrics['final_val_exact']:+.2%}")
    print(f"{'Val Within 3':<25} {equal_train_metrics['final_val_w3']:<20.2%} {he_train_metrics['final_val_w3']:<20.2%} {he_train_metrics['final_val_w3'] - equal_train_metrics['final_val_w3']:+.2%}")
    
    print("\n--- Test Set Metrics ---")
    print(f"{'Metric':<25} {'Equal Weights':<20} {'He/Kaiming':<20} {'Difference':<15}")
    print("-" * 70)
    print(f"{'MAE':<25} {equal_test_metrics['mae']:<20.4f} {he_test_metrics['mae']:<20.4f} {he_test_metrics['mae'] - equal_test_metrics['mae']:+.4f}")
    print(f"{'RMSE':<25} {equal_test_metrics['rmse']:<20.4f} {he_test_metrics['rmse']:<20.4f} {he_test_metrics['rmse'] - equal_test_metrics['rmse']:+.4f}")
    print(f"{'R²':<25} {equal_test_metrics['r2']:<20.4f} {he_test_metrics['r2']:<20.4f} {he_test_metrics['r2'] - equal_test_metrics['r2']:+.4f}")
    print(f"{'Exact Accuracy':<25} {equal_test_metrics['exact']:<20.2%} {he_test_metrics['exact']:<20.2%} {he_test_metrics['exact'] - equal_test_metrics['exact']:+.2%}")
    print(f"{'Within 1 Position':<25} {equal_test_metrics['within_1']:<20.2%} {he_test_metrics['within_1']:<20.2%} {he_test_metrics['within_1'] - equal_test_metrics['within_1']:+.2%}")
    print(f"{'Within 2 Positions':<25} {equal_test_metrics['within_2']:<20.2%} {he_test_metrics['within_2']:<20.2%} {he_test_metrics['within_2'] - equal_test_metrics['within_2']:+.2%}")
    print(f"{'Within 3 Positions':<25} {equal_test_metrics['within_3']:<20.2%} {he_test_metrics['within_3']:<20.2%} {he_test_metrics['within_3'] - equal_test_metrics['within_3']:+.2%}")
    
    # Determine winner
    print("\n" + "=" * 70)
    print("WINNER ANALYSIS")
    print("=" * 70)
    
    equal_wins = 0
    he_wins = 0
    
    metrics_to_compare = [
        ('MAE', 'lower', equal_test_metrics['mae'], he_test_metrics['mae']),
        ('RMSE', 'lower', equal_test_metrics['rmse'], he_test_metrics['rmse']),
        ('R²', 'higher', equal_test_metrics['r2'], he_test_metrics['r2']),
        ('Exact Accuracy', 'higher', equal_test_metrics['exact'], he_test_metrics['exact']),
        ('Within 1', 'higher', equal_test_metrics['within_1'], he_test_metrics['within_1']),
        ('Within 2', 'higher', equal_test_metrics['within_2'], he_test_metrics['within_2']),
        ('Within 3', 'higher', equal_test_metrics['within_3'], he_test_metrics['within_3']),
    ]
    
    print(f"\n{'Metric':<20} {'Better Method':<20}")
    print("-" * 40)
    for metric_name, direction, equal_val, he_val in metrics_to_compare:
        if direction == 'lower':
            better = 'Equal Weights' if equal_val < he_val else 'He/Kaiming'
            if equal_val < he_val:
                equal_wins += 1
            else:
                he_wins += 1
        else:
            better = 'Equal Weights' if equal_val > he_val else 'He/Kaiming'
            if equal_val > he_val:
                equal_wins += 1
            else:
                he_wins += 1
        print(f"{metric_name:<20} {better:<20}")
    
    print(f"\n{'='*70}")
    print(f"Final Score: Equal Weights {equal_wins} - {he_wins} He/Kaiming")
    if he_wins > equal_wins:
        print("He/Kaiming initialization WINS!")
    elif equal_wins > he_wins:
        print("Equal Weights initialization WINS!")
    else:
        print("TIE!")
    print("=" * 70)


if __name__ == '__main__':
    main()

