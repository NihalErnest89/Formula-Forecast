"""
Compare baseline model (2022-2024) vs expanded dataset (2020-2024).
This script trains both models and compares their performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import subprocess
import shutil
import pickle

# Import from top20/train.py
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import importlib.util
spec = importlib.util.spec_from_file_location("train_module", parent_dir / "top20" / "train.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

# Import the functions we need
F1Dataset = train_module.F1Dataset
F1NeuralNetwork = train_module.F1NeuralNetwork
load_data = train_module.load_data
prepare_features_and_labels = train_module.prepare_features_and_labels
train_epoch = train_module.train_epoch
evaluate_model = train_module.evaluate_model


def collect_data_with_years(training_years, test_year=2025, use_existing=True):
    """Collect data with specified training years."""
    print(f"\n{'='*70}")
    print(f"Collecting data with training years: {training_years}")
    print(f"{'='*70}")
    
    # Check if data already exists
    data_dir = parent_dir / 'data'
    year_key = "_".join(map(str, training_years))
    training_path = data_dir / f'training_data_{year_key}.csv'
    test_path = data_dir / f'test_data_{year_key}.csv'
    
    if use_existing and training_path.exists() and test_path.exists():
        print(f"  Loading existing data files...")
        training_df = pd.read_csv(training_path)
        test_df = pd.read_csv(test_path)
        print(f"  Loaded training data: {len(training_df)} samples")
        print(f"  Loaded test data: {len(test_df)} samples")
        return training_df, test_df, data_dir
    
    # Import collect_data functions
    print(f"  Collecting new data (this may take a while)...")
    collect_data_path = parent_dir / 'collect_data.py'
    spec = importlib.util.spec_from_file_location("collect_data", collect_data_path)
    collect_data_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collect_data_module)
    
    # Call organize_data
    training_df, test_df = collect_data_module.organize_data(training_years, test_year)
    
    # Save to data directory
    data_dir.mkdir(exist_ok=True, parents=True)
    training_df.to_csv(training_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  Saved training data: {len(training_df)} samples")
    print(f"  Saved test data: {len(test_df)} samples")
    
    return training_df, test_df, data_dir


def train_and_evaluate_model(training_df, test_df, config_name, device):
    """Train a model and evaluate it."""
    print(f"\n{'='*70}")
    print(f"Training {config_name} model...")
    print(f"{'='*70}")
    
    # Prepare training data
    print("\nPreparing training data (top 10 positions only)...")
    X_train, y_train, _, train_stats, feature_names = prepare_features_and_labels(
        training_df, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=True
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {len(feature_names)}")
    
    # Time-based validation split
    val_year = training_df['Year'].max()
    train_df_split = training_df[training_df['Year'] < val_year].copy()
    val_df_split = training_df[training_df['Year'] == val_year].copy()
    
    print(f"\n  Train years: {sorted(train_df_split['Year'].unique())}")
    print(f"  Validation year: {sorted(val_df_split['Year'].unique())}")
    
    X_train_split, y_train_split, _, _, _ = prepare_features_and_labels(
        train_df_split, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=True
    )
    X_val_split, y_val_split, _, _, _ = prepare_features_and_labels(
        val_df_split, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=True
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val_split)
    
    # Prepare test data
    print("\nPreparing test data...")
    X_test, y_test, _, test_stats, _ = prepare_features_and_labels(
        test_df, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=True
    )
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Test samples: {len(X_test)}")
    
    # Create datasets
    train_dataset = F1Dataset(X_train_scaled, y_train_split)
    val_dataset = F1Dataset(X_val_scaled, y_val_split)
    test_dataset = F1Dataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    input_size = X_train_scaled.shape[1]
    model = F1NeuralNetwork(
        input_size=input_size,
        hidden_sizes=[128, 64, 32],
        dropout_rate=0.4,
        equal_init=False
    ).to(device)
    
    # Loss and optimizer
    PositionAwareLoss = train_module.PositionAwareLoss
    criterion = PositionAwareLoss(exact_weight=5.0, base_loss='huber', delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
    )
    
    # Training loop
    print(f"\nTraining model...")
    best_val_mae = float('inf')
    best_model_state = None
    patience = 0
    max_patience = 30
    
    for epoch in range(200):
        # Train
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_mae, val_rmse, val_r2, val_exact, val_w1, val_w2, val_w3, _, _ = evaluate_model(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Train MAE={train_mae:.3f}, Val MAE={val_mae:.3f}, Val Exact={val_exact:.1f}%")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    print(f"\nEvaluating on test set...")
    criterion_eval = nn.MSELoss()
    test_loss, test_mae, test_rmse, test_r2, test_exact, test_w1, test_w2, test_w3, test_preds, test_labels = evaluate_model(
        model, test_loader, criterion_eval, device
    )
    
    results = {
        'config_name': config_name,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'validation': {
            'mae': best_val_mae,
            'rmse': val_rmse,
            'r2': val_r2,
            'exact': val_exact,
            'within_1': val_w1,
            'within_2': val_w2,
            'within_3': val_w3
        },
        'test': {
            'mae': test_mae,
            'rmse': test_rmse,
            'r2': test_r2,
            'exact': test_exact,
            'within_1': test_w1,
            'within_2': test_w2,
            'within_3': test_w3
        }
    }
    
    print(f"\n  Validation Results:")
    print(f"    MAE: {best_val_mae:.3f}, Exact: {val_exact:.1f}%, Within 3: {val_w3:.1f}%")
    print(f"  Test Results:")
    print(f"    MAE: {test_mae:.3f}, Exact: {test_exact:.1f}%, Within 3: {test_w3:.1f}%")
    
    return results, model, scaler


def main():
    """Main comparison function."""
    print("="*70)
    print("DATASET SIZE COMPARISON: Baseline (2022-2024) vs Expanded (2020-2024)")
    print("="*70)
    
    device = torch.device('cpu')
    print(f"\nUsing device: {device}")
    
    # Configuration 1: Baseline (2022-2024)
    baseline_years = [2022, 2023, 2024]
    baseline_name = "Baseline (2022-2024)"
    
    # Configuration 2: Expanded (2020-2024)
    expanded_years = [2020, 2021, 2022, 2023, 2024]
    expanded_name = "Expanded (2020-2024)"
    
    results = {}
    
    # Train baseline model
    print(f"\n{'='*70}")
    print("STEP 1: BASELINE MODEL (2022-2024)")
    print(f"{'='*70}")
    
    baseline_training_df, baseline_test_df, _ = collect_data_with_years(baseline_years, use_existing=True)
    baseline_results, baseline_model, baseline_scaler = train_and_evaluate_model(
        baseline_training_df, baseline_test_df, baseline_name, device
    )
    results['baseline'] = baseline_results
    
    # Train expanded model
    print(f"\n{'='*70}")
    print("STEP 2: EXPANDED MODEL (2020-2024)")
    print(f"{'='*70}")
    
    expanded_training_df, expanded_test_df, _ = collect_data_with_years(expanded_years, use_existing=True)
    expanded_results, expanded_model, expanded_scaler = train_and_evaluate_model(
        expanded_training_df, expanded_test_df, expanded_name, device
    )
    results['expanded'] = expanded_results
    
    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    
    print(f"\nTraining Data Size:")
    print(f"  Baseline: {results['baseline']['training_samples']} samples")
    print(f"  Expanded:  {results['expanded']['training_samples']} samples")
    print(f"  Difference: +{results['expanded']['training_samples'] - results['baseline']['training_samples']} samples")
    
    print(f"\nValidation Set Performance:")
    print(f"{'Metric':<20} {'Baseline':<20} {'Expanded':<20} {'Difference':<20}")
    print(f"{'-'*80}")
    for metric in ['mae', 'rmse', 'r2', 'exact', 'within_1', 'within_2', 'within_3']:
        baseline_val = results['baseline']['validation'][metric]
        expanded_val = results['expanded']['validation'][metric]
        diff = expanded_val - baseline_val
        diff_str = f"{diff:+.3f}" if isinstance(diff, float) else f"{diff:+.1f}%"
        print(f"{metric.capitalize():<20} {baseline_val:<20.3f} {expanded_val:<20.3f} {diff_str:<20}")
    
    print(f"\nTest Set Performance:")
    print(f"{'Metric':<20} {'Baseline':<20} {'Expanded':<20} {'Difference':<20}")
    print(f"{'-'*80}")
    for metric in ['mae', 'rmse', 'r2', 'exact', 'within_1', 'within_2', 'within_3']:
        baseline_test = results['baseline']['test'][metric]
        expanded_test = results['expanded']['test'][metric]
        diff = expanded_test - baseline_test
        diff_str = f"{diff:+.3f}" if isinstance(diff, float) else f"{diff:+.1f}%"
        print(f"{metric.capitalize():<20} {baseline_test:<20.3f} {expanded_test:<20.3f} {diff_str:<20}")
    
    # Save results
    results_path = parent_dir / 'top10' / 'dataset_comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    test_mae_diff = results['expanded']['test']['mae'] - results['baseline']['test']['mae']
    test_exact_diff = results['expanded']['test']['exact'] - results['baseline']['test']['exact']
    
    if test_mae_diff < 0:
        print(f"✓ Expanded dataset IMPROVES test MAE by {abs(test_mae_diff):.3f} positions")
    else:
        print(f"✗ Expanded dataset WORSENS test MAE by {test_mae_diff:.3f} positions")
    
    if test_exact_diff > 0:
        print(f"✓ Expanded dataset IMPROVES exact match accuracy by {test_exact_diff:.1f}%")
    else:
        print(f"✗ Expanded dataset WORSENS exact match accuracy by {abs(test_exact_diff):.1f}%")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()

