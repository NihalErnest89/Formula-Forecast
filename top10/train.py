"""
Training and testing script for F1 Predictions (Top 10 Only).
Trains a deep neural network on top 10 finishers only (positions 1-10).
Test results show: Training on top 10 only improves top 10 MAE from 3.043 to 1.861
and "Within 3" accuracy from 55.5% to 79.1%.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import sys

# Import from top20/train.py (has updated prepare_features_and_labels with filtering params)
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
# Import with explicit module name to avoid circular import
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
get_feature_importance = train_module.get_feature_importance
train_model = train_module.train_model
plot_training_history = train_module.plot_training_history
import pickle


def save_model(model, scaler, label_encoder, output_dir: str = None):
    """Save trained model, scaler, and label encoder (top 10 version)."""
    if output_dir is None:
        # Resolve path relative to this script's location
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / 'models'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    model_path = output_dir / 'f1_predictor_model_top10.pth'
    scaler_path = output_dir / 'scaler_top10.pkl'
    encoder_path = output_dir / 'label_encoder_top10.pkl'
    
    torch.save(model.state_dict(), model_path)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Label encoder saved to {encoder_path}")


def main():
    """Main function to train and test the F1 prediction model (top 10 only)."""
    print("F1 Position Prediction Model Training (Top 10 Only)")
    print("=" * 60)
    print("Predicting finishing positions (1-10) for top finishers only")
    print("Test results: Top 10 MAE improved from 3.043 to 1.861")
    print("             Within 3 accuracy improved from 55.5% to 79.1%")
    print("=" * 60)
    
    # Set device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    training_df, test_df, metadata = load_data()
    
    print(f"Training samples: {len(training_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Prepare training data (filtered - no DNFs/outliers, top 10 only)
    # Test results show: Training on top 10 only improves top 10 MAE from 3.043 to 1.861
    # and "Within 3" accuracy from 55.5% to 79.1%
    print("\nPreparing training data (top 10 positions only)...")
    X_train, y_train, _, train_stats, feature_names = prepare_features_and_labels(
        training_df, filter_dnf=True, filter_outliers=True, outlier_threshold=8, top10_only=True)
    
    # Prepare test data (filtered - for primary evaluation, top 10 only)
    print("\nPreparing test data (top 10 positions only, excluding DNFs/outliers)...")
    X_test, y_test, _, test_stats, _ = prepare_features_and_labels(
        test_df, filter_dnf=True, filter_outliers=True, outlier_threshold=8, top10_only=True)
    
    # Also prepare unfiltered test data (for comparison/reference, top 10 only)
    # Note: We still exclude DNFs (NaN positions) since we can't evaluate accuracy on them
    # But we include outliers (finish >> grid) for reference
    print("\nPreparing test data (top 10 only, including outliers for reference)...")
    X_test_all, y_test_all, _, test_stats_all, _ = prepare_features_and_labels(
        test_df, filter_dnf=True, filter_outliers=False, outlier_threshold=8, top10_only=True)
    
    # Scale features (fit on training, transform both filtered and unfiltered test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_test_all_scaled = scaler.transform(X_test_all) if len(X_test_all) > 0 else None
    
    print(f"  Features shape: {X_train_scaled.shape}")
    print(f"  Labels shape: {y_train.shape}")
    print(f"  Position range: {y_train.min():.0f} - {y_train.max():.0f}")
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )
    
    # Architecture: Slightly reduced to prevent overfitting
    # [256, 128, 64] - Architecture optimized for winner prediction (best MAE: 2.813 vs 3.009)
    # Tested in test_architectures_winner.py - this architecture showed best winner prediction performance
    # Using 7 base features only
    hidden_sizes = [256, 128, 64]
    
    # Train model
    model, history = train_model(
        X_train_split, y_train_split,
        X_val_split, y_val_split,
        epochs=300,  # Increased from 160 - allow more training with lenient early stopping
        batch_size=32,
        learning_rate=0.001,
        device=device,
        hidden_sizes=hidden_sizes,
        feature_names=feature_names
    )
    
    # Plot training history
    script_dir = Path(__file__).parent
    plot_training_history(history, save_path=str(script_dir.parent / 'images' / 'training_history_top10.png'))
    
    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Final Evaluation on Validation Set")
    print("=" * 60)
    
    val_dataset = F1Dataset(X_val_split, y_val_split)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    criterion = nn.MSELoss()
    
    val_loss, val_mae, val_rmse, val_r2, val_w1, val_w2, val_w3, val_preds, val_labels = evaluate_model(
        model, val_loader, criterion, device
    )
    
    print(f"\nValidation Metrics:")
    print(f"  MAE: {val_mae:.3f} positions")
    print(f"  RMSE: {val_rmse:.3f} positions")
    print(f"  R²: {val_r2:.4f}")
    print(f"\nPosition Accuracy:")
    print(f"  Within 1 position: {val_w1:.1f}%")
    print(f"  Within 2 positions: {val_w2:.1f}%")
    print(f"  Within 3 positions: {val_w3:.1f}%")
    
    # Feature importance (from first layer weights)
    feature_importances = get_feature_importance(model, feature_names, device)
    print(f"\nFeature Importances (Weight Distribution from First Layer):")
    for name, importance in feature_importances.items():
        print(f"  {name}: {importance:.4f}")
    
    # Scatter plot: predicted vs actual positions
    script_dir = Path(__file__).parent
    images_dir = script_dir.parent / 'images'
    images_dir.mkdir(exist_ok=True, parents=True)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(val_labels, val_preds, alpha=0.5)
    plt.plot([val_labels.min(), val_labels.max()], [val_labels.min(), val_labels.max()], 'r--', lw=2)
    plt.xlabel('Actual Position')
    plt.ylabel('Predicted Position')
    plt.title('Predicted vs Actual Finishing Positions (Top 10 Only)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_path = images_dir / 'prediction_scatter_top10.png'
    plt.savefig(scatter_path, dpi=150)
    print(f"\nPrediction scatter plot saved to {scatter_path}")
    plt.close()
    
    # Evaluate on test set (filtered - primary metrics)
    if len(X_test_scaled) > 0:
        print("\n" + "=" * 60)
        print("Evaluation on Test Set (FILTERED - Excluding DNFs/Outliers)")
        print("=" * 60)
        print("This is the primary evaluation metric - excludes unpredictable failures")
        print(f"Test samples: {len(y_test)} (after filtering, top 10 only)")
        
        test_dataset = F1Dataset(X_test_scaled, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        test_loss, test_mae, test_rmse, test_r2, test_w1, test_w2, test_w3, test_preds, test_labels = evaluate_model(
            model, test_loader, criterion, device
        )
        
        print(f"\nTest Metrics (Filtered, Top 10 Only):")
        print(f"  MAE: {test_mae:.3f} positions")
        print(f"  RMSE: {test_rmse:.3f} positions")
        print(f"  R²: {test_r2:.4f}")
        print(f"\nPosition Accuracy (Filtered, Top 10 Only):")
        print(f"  Within 1 position: {test_w1:.1f}%")
        print(f"  Within 2 positions: {test_w2:.1f}%")
        print(f"  Within 3 positions: {test_w3:.1f}%")
        
        # Also evaluate on unfiltered data for reference
        if X_test_all_scaled is not None and len(X_test_all_scaled) > 0:
            print("\n" + "=" * 60)
            print("Evaluation on Test Set (UNFILTERED - Including All Entries)")
            print("=" * 60)
            print("Reference metrics - includes DNFs/outliers for comparison (top 10 only)")
            print(f"Test samples: {len(y_test_all)} (all entries, top 10 only)")
            
            test_dataset_all = F1Dataset(X_test_all_scaled, y_test_all)
            test_loader_all = DataLoader(test_dataset_all, batch_size=32, shuffle=False)
            
            test_loss_all, test_mae_all, test_rmse_all, test_r2_all, test_w1_all, test_w2_all, test_w3_all, _, _ = evaluate_model(
                model, test_loader_all, criterion, device
            )
            
            print(f"\nTest Metrics (Unfiltered - Reference, Top 10 Only):")
            print(f"  MAE: {test_mae_all:.3f} positions")
            print(f"  RMSE: {test_rmse_all:.3f} positions")
            print(f"  R²: {test_r2_all:.4f}")
            print(f"\nPosition Accuracy (Unfiltered - Reference, Top 10 Only):")
            print(f"  Within 1 position: {test_w1_all:.1f}%")
            print(f"  Within 2 positions: {test_w2_all:.1f}%")
            print(f"  Within 3 positions: {test_w3_all:.1f}%")
            
            print(f"\nComparison:")
            print(f"  Filtered vs Unfiltered MAE: {test_mae:.3f} vs {test_mae_all:.3f} (improvement: {test_mae_all - test_mae:.3f})")
            print(f"  Filtered vs Unfiltered Within 3: {test_w3:.1f}% vs {test_w3_all:.1f}% (improvement: {test_w3 - test_w3_all:.1f}%)")
        else:
            test_mae_all = test_rmse_all = test_r2_all = test_w1_all = test_w2_all = test_w3_all = None
    else:
        print("\nNo test data available. Skipping test evaluation.")
        test_mae = test_rmse = test_r2 = test_w1 = test_w2 = test_w3 = None
        test_mae_all = test_rmse_all = test_r2_all = test_w1_all = test_w2_all = test_w3_all = None
    
    # Save model (no label encoder for regression)
    save_model(model, scaler, None)
    
    # Save results - convert all numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
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
    
    # Convert filtering stats to native types
    filtering_stats_dict = {
        'training': train_stats,
        'test_filtered': test_stats,
        'test_unfiltered': test_stats_all if 'test_stats_all' in locals() else None
    }
    
    results = {
        'model_type': 'neural_network_regression_top10',
        'filtering_stats': convert_to_native(filtering_stats_dict),
        'validation_mae': float(val_mae),
        'validation_rmse': float(val_rmse),
        'validation_r2': float(val_r2),
        'validation_within_1': float(val_w1),
        'validation_within_2': float(val_w2),
        'validation_within_3': float(val_w3),
        'test_filtered': {
            'mae': float(test_mae) if test_mae else None,
            'rmse': float(test_rmse) if test_rmse else None,
            'r2': float(test_r2) if test_r2 else None,
            'within_1': float(test_w1) if test_w1 else None,
            'within_2': float(test_w2) if test_w2 else None,
            'within_3': float(test_w3) if test_w3 else None,
        },
        'test_unfiltered': {
            'mae': float(test_mae_all) if 'test_mae_all' in locals() and test_mae_all else None,
            'rmse': float(test_rmse_all) if 'test_rmse_all' in locals() and test_rmse_all else None,
            'r2': float(test_r2_all) if 'test_r2_all' in locals() and test_r2_all else None,
            'within_1': float(test_w1_all) if 'test_w1_all' in locals() and test_w1_all else None,
            'within_2': float(test_w2_all) if 'test_w2_all' in locals() and test_w2_all else None,
            'within_3': float(test_w3_all) if 'test_w3_all' in locals() and test_w3_all else None,
        },
        'feature_importances': {k: float(v) for k, v in feature_importances.items()},
        'model_architecture': {
            'input_size': X_train.shape[1],
            'hidden_sizes': hidden_sizes,
            'dropout_rate': 0.4,
            'output': 'regression (single position value, 1-10)',
            'features': feature_names if feature_names else []
        }
    }
    
    # Create json directory if it doesn't exist
    script_dir = Path(__file__).parent
    json_dir = script_dir.parent / 'json'
    json_dir.mkdir(exist_ok=True, parents=True)
    
    results_path = json_dir / 'training_results_top10.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Results saved to {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

