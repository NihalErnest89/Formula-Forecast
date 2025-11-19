"""
Test different neural network architectures focusing on winner prediction accuracy.
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

def test_architectures():
    print("F1 Position Prediction - Architecture Comparison (Winner Focus)")
    print("=" * 70)
    print("Testing different architectures with focus on winner prediction accuracy")
    print("Using TOP 10 training method (positions 1-10 only)")
    print("=" * 70)

    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load data (using parent directory path resolution)
    print("\nLoading data...")
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    training_df, test_df, _ = load_data(data_dir=str(data_dir))

    print(f"Training samples: {len(training_df)}")
    print(f"Test samples: {len(test_df)}")

    # Prepare training data (top 10 only for better winner prediction)
    print("\nPreparing training data (top 10 positions only)...")
    X_train, y_train, _, train_stats, feature_names = prepare_features_and_labels(
        training_df, filter_dnf=True, filter_outliers=True, outlier_threshold=8, top10_only=True)
    
    X_test, y_test, _, test_stats, _ = prepare_features_and_labels(
        test_df, filter_dnf=True, filter_outliers=True, outlier_threshold=8, top10_only=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    print(f"  Training on: {len(X_train)} samples (positions 1-10)")
    print(f"  Position range: {y_train.min():.0f} - {y_train.max():.0f}")

    # Architectures to test
    architectures = {
        'current': [192, 96, 48],
        'deeper': [192, 96, 48, 24],
        'wider': [256, 128, 64],
        'wider_deeper': [256, 128, 64, 32],
        'very_wide': [384, 192, 96],
        'very_deep': [128, 96, 64, 48, 32]
    }

    results = {}

    for arch_name, hidden_sizes in architectures.items():
        print("\n" + "="*70)
        print(f"TESTING: {arch_name.upper()} - {hidden_sizes}")
        print("="*70)
        
        # Train model
        model, history = train_model(
            X_train_split, y_train_split,
            X_val_split, y_val_split,
            epochs=150,  # Reduced for faster testing
            batch_size=32,
            learning_rate=0.001,
            device=device,
            hidden_sizes=hidden_sizes,
            feature_names=feature_names,
            early_stop_patience=30
        )

        # Evaluate on test set
        test_dataset = F1Dataset(X_test_scaled, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        criterion = nn.HuberLoss(delta=1.0)

        test_loss, test_mae, test_rmse, test_r2, test_w1, test_w2, test_w3, test_preds, test_labels = evaluate_model(
            model, test_loader, criterion, device
        )

        # Winner prediction accuracy (position 1)
        winner_mask = y_test == 1
        if winner_mask.sum() > 0:
            X_test_winners = X_test_scaled[winner_mask]
            y_test_winners = y_test[winner_mask]
            
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test_winners).to(device)
                winner_preds = model(X_tensor).cpu().numpy().flatten()
                # Clip to 1-10 range to match training data
                winner_preds = np.clip(winner_preds, 1, 10)
            
            winner_mae = np.mean(np.abs(winner_preds - y_test_winners))
            winner_rmse = np.sqrt(np.mean((winner_preds - y_test_winners)**2))
            winner_within_1 = (np.abs(winner_preds - y_test_winners) <= 1).sum() / len(winner_preds) * 100
            winner_within_2 = (np.abs(winner_preds - y_test_winners) <= 2).sum() / len(winner_preds) * 100
            winner_exact = (np.abs(winner_preds - y_test_winners) < 0.5).sum() / len(winner_preds) * 100
            
            # Also check if winner is in top 3 predicted
            # Get all predictions for races with winners
            all_preds = []
            all_labels = []
            for features, labels in test_loader:
                features = features.to(device)
                with torch.no_grad():
                    preds = model(features).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
            
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            # Clip all predictions to 1-10 range to match training data
            all_preds = np.clip(all_preds, 1, 10)
            
            # Group by race (assuming test data is organized by race)
            # For simplicity, check if winner (position 1) is in top 3 predictions per race
            winner_in_top3 = 0
            winner_in_top5 = 0
            total_races = 0
            
            # Simple approach: check if any position 1 has prediction <= 3
            winner_positions = all_preds[all_labels == 1]
            winner_in_top3 = (winner_positions <= 3).sum() / len(winner_positions) * 100 if len(winner_positions) > 0 else 0
            winner_in_top5 = (winner_positions <= 5).sum() / len(winner_positions) * 100 if len(winner_positions) > 0 else 0
        else:
            winner_mae = winner_rmse = winner_within_1 = winner_within_2 = winner_exact = winner_in_top3 = winner_in_top5 = None

        results[arch_name] = {
            'architecture': hidden_sizes,
            'total_params': sum(p.numel() for p in model.parameters()),
            'test_all': {
                'mae': float(test_mae),
                'rmse': float(test_rmse),
                'r2': float(test_r2),
                'within_1': float(test_w1),
                'within_2': float(test_w2),
                'within_3': float(test_w3),
            },
            'winner_prediction': {
                'mae': float(winner_mae) if winner_mae is not None else None,
                'rmse': float(winner_rmse) if winner_rmse is not None else None,
                'within_1': float(winner_within_1) if winner_within_1 is not None else None,
                'within_2': float(winner_within_2) if winner_within_2 is not None else None,
                'exact': float(winner_exact) if winner_exact is not None else None,
                'in_top3_predicted': float(winner_in_top3) if winner_in_top3 is not None else None,
                'in_top5_predicted': float(winner_in_top5) if winner_in_top5 is not None else None,
            }
        }

        print(f"\nResults for {arch_name} ({hidden_sizes}):")
        print(f"  Total parameters: {results[arch_name]['total_params']:,}")
        print(f"  Test MAE: {test_mae:.3f} positions")
        print(f"  Test Within 3: {test_w3:.1f}%")
        
        if winner_mae is not None:
            print(f"\n  Winner Prediction (Position 1):")
            print(f"    Winner MAE: {winner_mae:.3f} positions")
            print(f"    Winner RMSE: {winner_rmse:.3f} positions")
            print(f"    Winner Within 1: {winner_within_1:.1f}%")
            print(f"    Winner Within 2: {winner_within_2:.1f}%")
            print(f"    Winner Exact (<0.5): {winner_exact:.1f}%")
            print(f"    Winner in Top 3 Predicted: {winner_in_top3:.1f}%")
            print(f"    Winner in Top 5 Predicted: {winner_in_top5:.1f}%")

    # Comparison
    print("\n" + "="*70)
    print("ARCHITECTURE COMPARISON")
    print("="*70)
    
    print("\nOverall Test Performance:")
    print(f"{'Architecture':<20} {'Params':<12} {'MAE':<8} {'Within 3':<10}")
    print("-" * 60)
    for arch_name, result in results.items():
        print(f"{arch_name:<20} {result['total_params']:<12,} {result['test_all']['mae']:<8.3f} {result['test_all']['within_3']:<10.1f}%")
    
    print("\nWinner Prediction Performance (Position 1):")
    print(f"{'Architecture':<20} {'MAE':<8} {'Within 1':<10} {'Within 2':<10} {'Exact':<10} {'Top3':<10} {'Top5':<10}")
    print("-" * 90)
    for arch_name, result in results.items():
        wp = result['winner_prediction']
        if wp['mae'] is not None:
            print(f"{arch_name:<20} {wp['mae']:<8.3f} {wp['within_1']:<10.1f}% {wp['within_2']:<10.1f}% {wp['exact']:<10.1f}% {wp['in_top3_predicted']:<10.1f}% {wp['in_top5_predicted']:<10.1f}%")
        else:
            print(f"{arch_name:<20} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    # Find best architecture for winner prediction
    best_winner_mae = float('inf')
    best_winner_arch = None
    for arch_name, result in results.items():
        wp = result['winner_prediction']
        if wp['mae'] is not None and wp['mae'] < best_winner_mae:
            best_winner_mae = wp['mae']
            best_winner_arch = arch_name
    
    if best_winner_arch:
        print(f"\n[+] Best architecture for winner prediction: {best_winner_arch}")
        print(f"    Architecture: {results[best_winner_arch]['architecture']}")
        print(f"    Winner MAE: {results[best_winner_arch]['winner_prediction']['mae']:.3f}")
        print(f"    Winner Within 1: {results[best_winner_arch]['winner_prediction']['within_1']:.1f}%")
        print(f"    Winner Exact: {results[best_winner_arch]['winner_prediction']['exact']:.1f}%")

    # Save results
    with open(Path(JSON_DIR) / 'architecture_comparison_winner.json', 'w') as f:
        json.dump(convert_to_native(results), f, indent=2)
    
    print(f"\nResults saved to: {Path(JSON_DIR) / 'architecture_comparison_winner.json'}")

if __name__ == "__main__":
    test_architectures()

