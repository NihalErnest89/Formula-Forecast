"""
Evaluate all baseline models on the same top-10 filtered test set for fair comparison.
All models are evaluated on 2025 test data, filtered to top-10 positions only.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
import sys

# Add parent directory to path to import from top10 and top20
sys.path.insert(0, str(Path(__file__).parent))

from top10.config import FEATURE_COLS
from top20.train import (
    prepare_features_and_labels, 
    F1NeuralNetwork as Top20F1NeuralNetwork, 
    F1Dataset,
    load_data as load_data_top20,
    train_model as train_model_top20
)
from top10.train import F1NeuralNetwork as Top10F1NeuralNetwork


def load_test_data():
    """Load 2025 test data."""
    test_path = Path('data/test_data.csv')
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")
    return pd.read_csv(test_path)


def evaluate_top10_model():
    """Evaluate top10 regression model."""
    print("Evaluating Top-10 Regression Model...")
    
    # Load model
    model_path = Path('models/f1_predictor_model_top10.pth')
    scaler_path = Path('models/scaler_top10.pkl')
    
    if not model_path.exists() or not scaler_path.exists():
        print("  Top-10 model files not found, skipping...")
        return None
    
    device = torch.device('cpu')
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load test data
    test_df = load_test_data()
    
    # Prepare features (top10 only, filtered)
    X_test, y_test, _, _, _ = prepare_features_and_labels(
        test_df,
        filter_dnf=True,
        filter_outliers=True,
        outlier_threshold=6,
        top10_only=True
    )
    
    if len(X_test) == 0:
        print("  No test data available")
        return None
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Load model
    model = Top10F1NeuralNetwork(
        input_size=9,
        hidden_sizes=[128, 64, 32],
        dropout_rate=0.4,
        equal_init=False
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Make predictions
    test_dataset = F1Dataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    mae = np.mean(np.abs(all_labels - all_preds))
    rmse = np.sqrt(np.mean((all_labels - all_preds) ** 2))
    r2 = 1 - np.sum((all_labels - all_preds) ** 2) / np.sum((all_labels - np.mean(all_labels)) ** 2)
    
    position_error = np.abs(all_labels - all_preds)
    exact = np.mean(np.round(all_preds) == all_labels) * 100
    within_1 = np.mean(position_error <= 1) * 100
    within_2 = np.mean(position_error <= 2) * 100
    within_3 = np.mean(position_error <= 3) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'exact': exact,
        'within_1': within_1,
        'within_2': within_2,
        'within_3': within_3
    }


def evaluate_top20_model_on_top10():
    """Evaluate top20 model on top10 subset."""
    print("Evaluating Top-20 Model (on Top-10 subset)...")
    
    # Load model (check if top20 model exists)
    model_path = Path('models/f1_predictor_model.pth')
    scaler_path = Path('models/scaler.pkl')
    
    device = torch.device('cpu')
    
    # If model doesn't exist, we need to train it first
    if not model_path.exists() or not scaler_path.exists():
        print("  Top-20 model not found. Training top20 model now...")
        try:
            # Load training data
            training_df, _, _ = load_data_top20()
            
            # Prepare training data (all positions 1-20, filtered)
            X_train, y_train, _, _, feature_names = prepare_features_and_labels(
                training_df,
                filter_dnf=True,
                filter_outliers=True,
                outlier_threshold=6,
                top10_only=False  # Train on all positions
            )
            
            # Time-based split: use most recent year for validation
            val_year = training_df['Year'].max()
            train_df_split = training_df[training_df['Year'] < val_year].copy()
            val_df_split = training_df[training_df['Year'] == val_year].copy()
            
            if len(val_df_split) == 0:
                print("  Cannot perform time-based split, skipping top20 training...")
                return None
            
            X_train_split, y_train_split, _, _, _ = prepare_features_and_labels(
                train_df_split,
                filter_dnf=True,
                filter_outliers=True,
                outlier_threshold=6,
                top10_only=False
            )
            X_val_split, y_val_split, _, _, _ = prepare_features_and_labels(
                val_df_split,
                filter_dnf=True,
                filter_outliers=True,
                outlier_threshold=6,
                top10_only=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_split)
            X_val_scaled = scaler.transform(X_val_split)
            
            # Train model (quick training with fewer epochs for comparison)
            print("  Training top20 model (this may take a few minutes)...")
            model, _ = train_model_top20(
                X_train_scaled, y_train_split,
                X_val_scaled, y_val_split,
                epochs=50,  # Reduced epochs for faster training (just for comparison)
                batch_size=32,
                learning_rate=0.005,
                device=device,
                hidden_sizes=[256, 128, 64],
                feature_names=feature_names,
                early_stop_patience=10  # Early stopping to speed up
            )
            
            # Save model and scaler
            model_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), model_path)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print("  Top20 model trained and saved!")
            
        except Exception as e:
            print(f"  Could not train top20 model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Load scaler
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"  Could not load scaler: {e}")
        return None
    
    # Load test data
    test_df = load_test_data()
    
    # Prepare features (top10 only, filtered) - use same 9 features as top10
    X_test, y_test, _, _, _ = prepare_features_and_labels(
        test_df,
        filter_dnf=True,
        filter_outliers=True,
        outlier_threshold=6,
        top10_only=True
    )
    
    if len(X_test) == 0:
        print("  No test data available")
        return None
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Load model - top20 uses same architecture but trained on all positions
    model = Top20F1NeuralNetwork(
        input_size=9,  # Same 9 features
        hidden_sizes=[256, 128, 64],  # Top20 architecture
        dropout_rate=0.4,
        equal_init=False
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"  Could not load top20 model: {e}")
        return None
    
    # Make predictions
    test_dataset = F1Dataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    mae = np.mean(np.abs(all_labels - all_preds))
    rmse = np.sqrt(np.mean((all_labels - all_preds) ** 2))
    r2 = 1 - np.sum((all_labels - all_preds) ** 2) / np.sum((all_labels - np.mean(all_labels)) ** 2)
    
    position_error = np.abs(all_labels - all_preds)
    exact = np.mean(np.round(all_preds) == all_labels) * 100
    within_1 = np.mean(position_error <= 1) * 100
    within_2 = np.mean(position_error <= 2) * 100
    within_3 = np.mean(position_error <= 3) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'exact': exact,
        'within_1': within_1,
        'within_2': within_2,
        'within_3': within_3
    }


def evaluate_random_forest_on_top10():
    """Evaluate Random Forest on top10 subset with 9 features."""
    print("Evaluating Random Forest (on Top-10 subset with 9 features)...")
    
    # Load test data
    test_df = load_test_data()
    
    # Prepare features (top10 only, filtered) - use 9 features to match top10
    X_test, y_test, _, _, _ = prepare_features_and_labels(
        test_df,
        filter_dnf=True,
        filter_outliers=True,
        outlier_threshold=6,
        top10_only=True
    )
    
    if len(X_test) == 0:
        print("  No test data available")
        return None
    
    # Load training data to train RF model
    train_path = Path('data/training_data.csv')
    if not train_path.exists():
        print("  Training data not found, skipping RF evaluation...")
        return None
    
    train_df = pd.read_csv(train_path)
    
    # Prepare training features (top10 only, filtered) with 9 features
    X_train, y_train, _, _, _ = prepare_features_and_labels(
        train_df,
        filter_dnf=True,
        filter_outliers=True,
        outlier_threshold=6,
        top10_only=True
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    position_error = np.abs(y_test - y_pred)
    exact = np.mean(np.round(y_pred) == y_test) * 100
    within_1 = np.mean(position_error <= 1) * 100
    within_2 = np.mean(position_error <= 2) * 100
    within_3 = np.mean(position_error <= 3) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'exact': exact,
        'within_1': within_1,
        'within_2': within_2,
        'within_3': within_3
    }


def main():
    print("=" * 70)
    print("Baseline Comparison Evaluation")
    print("All models evaluated on 2025 test data, top-10 positions only")
    print("=" * 70)
    
    results = {}
    
    # Evaluate Top-10 model
    top10_results = evaluate_top10_model()
    if top10_results:
        results['Top-10 Regression'] = top10_results
    
    # Evaluate Top-20 model on top10
    top20_results = evaluate_top20_model_on_top10()
    if top20_results:
        results['Top-20 NN (top10 eval)'] = top20_results
    
    # Evaluate Random Forest on top10
    rf_results = evaluate_random_forest_on_top10()
    if rf_results:
        results['Random Forest (top10 eval)'] = rf_results
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    if not results:
        print("No results available. Check model files and data paths.")
        return
    
    print(f"\n{'Model':<30} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Exact':<8} {'≤1':<8} {'≤2':<8} {'≤3':<8}")
    print("-" * 90)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<30} {metrics['mae']:.3f}   {metrics['rmse']:.3f}   {metrics['r2']:.3f}   "
              f"{metrics['exact']:.1f}%   {metrics['within_1']:.1f}%   {metrics['within_2']:.1f}%   {metrics['within_3']:.1f}%")
    
    print("\nDetailed Metrics:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  Exact: {metrics['exact']:.2f}%")
        print(f"  Within 1: {metrics['within_1']:.2f}%")
        print(f"  Within 2: {metrics['within_2']:.2f}%")
        print(f"  Within 3: {metrics['within_3']:.2f}%")


if __name__ == '__main__':
    main()

