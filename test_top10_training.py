"""
Test if training only on top 10 finishers improves winner prediction accuracy.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from train import F1NeuralNetwork, prepare_features_and_labels, train_model, evaluate_model, F1Dataset, load_data

def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        try:
            if hasattr(obj, 'item'):
                return obj.item()
        except (AttributeError, ValueError):
            pass
        return obj

# --- Configuration ---
JSON_DIR = 'json'

Path(JSON_DIR).mkdir(exist_ok=True)

def test_top10_training():
    print("F1 Position Prediction - Top 10 Training Test")
    print("=" * 70)
    print("Testing if training only on top 10 finishers improves winner prediction")
    print("=" * 70)

    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    training_df, test_df, _ = load_data()

    print(f"Training samples: {len(training_df)}")
    print(f"Test samples: {len(test_df)}")

    # Test 1: Train on ALL drivers (current approach)
    print("\n" + "="*70)
    print("APPROACH 1: Train on ALL drivers (positions 1-20)")
    print("="*70)
    
    X_train_all, y_train_all, _, _, feature_names = prepare_features_and_labels(
        training_df, filter_dnf=True, filter_outliers=True, outlier_threshold=8)
    X_test_all, y_test_all, _, _, _ = prepare_features_and_labels(
        test_df, filter_dnf=True, filter_outliers=True, outlier_threshold=8)

    scaler_all = StandardScaler()
    X_train_scaled_all = scaler_all.fit_transform(X_train_all)
    X_test_scaled_all = scaler_all.transform(X_test_all)

    X_train_split_all, X_val_split_all, y_train_split_all, y_val_split_all = train_test_split(
        X_train_scaled_all, y_train_all, test_size=0.2, random_state=42
    )

    print(f"  Training on: {len(X_train_all)} samples (all positions 1-20)")
    print(f"  Position range: {y_train_all.min():.0f} - {y_train_all.max():.0f}")

    # Train model
    model_all, history_all = train_model(
        X_train_split_all, y_train_split_all,
        X_val_split_all, y_val_split_all,
        epochs=100,  # Reduced for faster testing
        batch_size=32,
        learning_rate=0.001,
        device=device,
        hidden_sizes=[192, 96, 48],
        feature_names=feature_names,
        early_stop_patience=20
    )

    # Evaluate on all test data
    test_dataset_all = F1Dataset(X_test_scaled_all, y_test_all)
    test_loader_all = DataLoader(test_dataset_all, batch_size=32, shuffle=False)
    criterion = nn.HuberLoss(delta=1.0)

    test_loss_all, test_mae_all, test_rmse_all, test_r2_all, test_w1_all, test_w2_all, test_w3_all, _, _ = evaluate_model(
        model_all, test_loader_all, criterion, device
    )

    # Evaluate on top 10 only (for comparison)
    top10_mask_test = y_test_all <= 10
    if top10_mask_test.sum() > 0:
        X_test_top10 = X_test_scaled_all[top10_mask_test]
        y_test_top10 = y_test_all[top10_mask_test]
        
        test_dataset_top10 = F1Dataset(X_test_top10, y_test_top10)
        test_loader_top10 = DataLoader(test_dataset_top10, batch_size=32, shuffle=False)
        
        test_loss_top10, test_mae_top10, test_rmse_top10, test_r2_top10, test_w1_top10, test_w2_top10, test_w3_top10, _, _ = evaluate_model(
            model_all, test_loader_top10, criterion, device
        )
        
        # Winner prediction accuracy (position 1)
        winner_mask = y_test_all == 1
        if winner_mask.sum() > 0:
            X_test_winners = X_test_scaled_all[winner_mask]
            y_test_winners = y_test_all[winner_mask]
            
            model_all.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test_winners).to(device)
                winner_preds = model_all(X_tensor).cpu().numpy().flatten()
            
            winner_mae = np.mean(np.abs(winner_preds - y_test_winners))
            winner_within_1 = (np.abs(winner_preds - y_test_winners) <= 1).sum() / len(winner_preds) * 100
            winner_within_2 = (np.abs(winner_preds - y_test_winners) <= 2).sum() / len(winner_preds) * 100
        else:
            winner_mae = winner_within_1 = winner_within_2 = None
    else:
        test_mae_top10 = test_w3_top10 = None
        winner_mae = winner_within_1 = winner_within_2 = None

    results_all = {
        'approach': 'train_on_all_20',
        'training_samples': len(X_train_all),
        'training_position_range': f"{y_train_all.min():.0f}-{y_train_all.max():.0f}",
        'test_all': {
            'mae': float(test_mae_all),
            'rmse': float(test_rmse_all),
            'r2': float(test_r2_all),
            'within_3': float(test_w3_all),
        },
        'test_top10_only': {
            'mae': float(test_mae_top10) if test_mae_top10 is not None else None,
            'within_3': float(test_w3_top10) if test_w3_top10 is not None else None,
        },
        'winner_prediction': {
            'mae': float(winner_mae) if winner_mae is not None else None,
            'within_1': float(winner_within_1) if winner_within_1 is not None else None,
            'within_2': float(winner_within_2) if winner_within_2 is not None else None,
        }
    }

    print(f"\nResults (trained on all 20, tested on all):")
    print(f"  Test MAE: {test_mae_all:.3f} positions")
    print(f"  Test Within 3: {test_w3_all:.1f}%")
    
    if test_mae_top10 is not None:
        print(f"\nResults (trained on all 20, tested on top 10 only):")
        print(f"  Test MAE (top 10): {test_mae_top10:.3f} positions")
        print(f"  Test Within 3 (top 10): {test_w3_top10:.1f}%")
    
    if winner_mae is not None:
        print(f"\nWinner Prediction (position 1) Accuracy:")
        print(f"  Winner MAE: {winner_mae:.3f} positions")
        print(f"  Winner Within 1: {winner_within_1:.1f}%")
        print(f"  Winner Within 2: {winner_within_2:.1f}%")

    # Test 2: Train on TOP 10 ONLY
    print("\n" + "="*70)
    print("APPROACH 2: Train on TOP 10 ONLY (positions 1-10)")
    print("="*70)
    
    # Filter training data to top 10 only
    training_df_top10 = training_df[
        (training_df['ActualPosition'] <= 10) & 
        (~training_df['IsDNF']) & 
        ((training_df['ActualPosition'] - training_df['GridPosition']) <= 8)
    ].copy()
    
    test_df_top10 = test_df[
        (test_df['ActualPosition'] <= 10) & 
        (~test_df['IsDNF']) & 
        ((test_df['ActualPosition'] - test_df['GridPosition']) <= 8)
    ].copy()
    
    X_train_top10, y_train_top10, _, _, _ = prepare_features_and_labels(
        training_df_top10, filter_dnf=False, filter_outliers=False, outlier_threshold=8)
    X_test_top10_only, y_test_top10_only, _, _, _ = prepare_features_and_labels(
        test_df_top10, filter_dnf=False, filter_outliers=False, outlier_threshold=8)

    scaler_top10 = StandardScaler()
    X_train_scaled_top10 = scaler_top10.fit_transform(X_train_top10)
    X_test_scaled_top10_only = scaler_top10.transform(X_test_top10_only)

    X_train_split_top10, X_val_split_top10, y_train_split_top10, y_val_split_top10 = train_test_split(
        X_train_scaled_top10, y_train_top10, test_size=0.2, random_state=42
    )

    print(f"  Training on: {len(X_train_top10)} samples (positions 1-10 only)")
    print(f"  Position range: {y_train_top10.min():.0f} - {y_train_top10.max():.0f}")

    # Train model
    model_top10, history_top10 = train_model(
        X_train_split_top10, y_train_split_top10,
        X_val_split_top10, y_val_split_top10,
        epochs=100,  # Reduced for faster testing
        batch_size=32,
        learning_rate=0.001,
        device=device,
        hidden_sizes=[192, 96, 48],
        feature_names=feature_names,
        early_stop_patience=20
    )

    # Evaluate on top 10 test data
    test_dataset_top10_only = F1Dataset(X_test_scaled_top10_only, y_test_top10_only)
    test_loader_top10_only = DataLoader(test_dataset_top10_only, batch_size=32, shuffle=False)

    test_loss_top10_only, test_mae_top10_only, test_rmse_top10_only, test_r2_top10_only, test_w1_top10_only, test_w2_top10_only, test_w3_top10_only, _, _ = evaluate_model(
        model_top10, test_loader_top10_only, criterion, device
    )
    
    # Winner prediction accuracy (position 1)
    winner_mask_top10 = y_test_top10_only == 1
    if winner_mask_top10.sum() > 0:
        X_test_winners_top10 = X_test_scaled_top10_only[winner_mask_top10]
        y_test_winners_top10 = y_test_top10_only[winner_mask_top10]
        
        model_top10.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test_winners_top10).to(device)
            winner_preds_top10 = model_top10(X_tensor).cpu().numpy().flatten()
        
        winner_mae_top10 = np.mean(np.abs(winner_preds_top10 - y_test_winners_top10))
        winner_within_1_top10 = (np.abs(winner_preds_top10 - y_test_winners_top10) <= 1).sum() / len(winner_preds_top10) * 100
        winner_within_2_top10 = (np.abs(winner_preds_top10 - y_test_winners_top10) <= 2).sum() / len(winner_preds_top10) * 100
    else:
        winner_mae_top10 = winner_within_1_top10 = winner_within_2_top10 = None

    results_top10 = {
        'approach': 'train_on_top10_only',
        'training_samples': len(X_train_top10),
        'training_position_range': f"{y_train_top10.min():.0f}-{y_train_top10.max():.0f}",
        'test_top10_only': {
            'mae': float(test_mae_top10_only),
            'rmse': float(test_rmse_top10_only),
            'r2': float(test_r2_top10_only),
            'within_3': float(test_w3_top10_only),
        },
        'winner_prediction': {
            'mae': float(winner_mae_top10) if winner_mae_top10 is not None else None,
            'within_1': float(winner_within_1_top10) if winner_within_1_top10 is not None else None,
            'within_2': float(winner_within_2_top10) if winner_within_2_top10 is not None else None,
        }
    }

    print(f"\nResults (trained on top 10, tested on top 10):")
    print(f"  Test MAE: {test_mae_top10_only:.3f} positions")
    print(f"  Test Within 3: {test_w3_top10_only:.1f}%")
    
    if winner_mae_top10 is not None:
        print(f"\nWinner Prediction (position 1) Accuracy:")
        print(f"  Winner MAE: {winner_mae_top10:.3f} positions")
        print(f"  Winner Within 1: {winner_within_1_top10:.1f}%")
        print(f"  Winner Within 2: {winner_within_2_top10:.1f}%")

    # Comparison
    print(f"\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    print(f"\nTop 10 Test Set Performance:")
    if test_mae_top10 is not None and test_mae_top10_only is not None:
        mae_diff = test_mae_top10 - test_mae_top10_only
        w3_diff = test_w3_top10_only - test_w3_top10
        print(f"  All-20 training: MAE={test_mae_top10:.3f}, Within 3={test_w3_top10:.1f}%")
        print(f"  Top-10 training: MAE={test_mae_top10_only:.3f}, Within 3={test_w3_top10_only:.1f}%")
        print(f"  Difference: MAE={mae_diff:+.3f}, Within 3={w3_diff:+.1f}%")
        if mae_diff > 0 and w3_diff > 0:
            print(f"  [+] Top-10 training is BETTER!")
        elif mae_diff < 0 and w3_diff < 0:
            print(f"  [-] All-20 training is BETTER")
        else:
            print(f"  [~] Mixed results")
    
    print(f"\nWinner Prediction (Position 1) Performance:")
    if winner_mae is not None and winner_mae_top10 is not None:
        winner_mae_diff = winner_mae - winner_mae_top10
        winner_w1_diff = winner_within_1_top10 - winner_within_1
        print(f"  All-20 training: MAE={winner_mae:.3f}, Within 1={winner_within_1:.1f}%")
        print(f"  Top-10 training: MAE={winner_mae_top10:.3f}, Within 1={winner_within_1_top10:.1f}%")
        print(f"  Difference: MAE={winner_mae_diff:+.3f}, Within 1={winner_w1_diff:+.1f}%")
        if winner_mae_diff > 0 and winner_w1_diff > 0:
            print(f"  [+] Top-10 training is BETTER for winners!")
        elif winner_mae_diff < 0 and winner_w1_diff < 0:
            print(f"  [-] All-20 training is BETTER for winners")
        else:
            print(f"  [~] Mixed results for winners")

    # Save results
    all_results = {
        'train_on_all_20': results_all,
        'train_on_top10_only': results_top10
    }
    
    with open(Path(JSON_DIR) / 'top10_training_test.json', 'w') as f:
        json.dump(convert_to_native(all_results), f, indent=2)
    
    print(f"\nResults saved to: {Path(JSON_DIR) / 'top10_training_test.json'}")

if __name__ == "__main__":
    test_top10_training()

