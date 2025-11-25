"""
Compare Classification vs Regression models for F1 position prediction.
Tests both models on the same test set and compares metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
from scipy.optimize import linear_sum_assignment
import pickle
import sys

# Import shared utilities
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))
import importlib.util

# Import regression model utilities
spec_reg = importlib.util.spec_from_file_location("train_module", parent_dir / "top20" / "train.py")
train_module = importlib.util.module_from_spec(spec_reg)
spec_reg.loader.exec_module(train_module)

load_data = train_module.load_data
prepare_features_and_labels = train_module.prepare_features_and_labels

# Import classification model
spec_cls = importlib.util.spec_from_file_location("cls_train", parent_dir / "top10_classification" / "train.py")
cls_train_module = importlib.util.module_from_spec(spec_cls)
spec_cls.loader.exec_module(cls_train_module)

F1ClassificationNetwork = cls_train_module.F1ClassificationNetwork
F1ClassificationDataset = cls_train_module.F1ClassificationDataset


def load_regression_model():
    """Load regression model (ensemble)."""
    model_dir = parent_dir / 'models'
    scaler_path = model_dir / 'scaler_top10.pkl'
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load ensemble models
    models = []
    device = torch.device('cpu')
    input_size = getattr(scaler, 'n_features_in_', 9)
    
    for i in range(3):
        model_path = model_dir / f'f1_predictor_model_top10_ensemble_{i}.pth'
        if model_path.exists():
            from top10.predict import F1NeuralNetwork
            model = F1NeuralNetwork(
                input_size=input_size,
                hidden_sizes=[128, 64, 32],
                dropout_rate=0.4
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models.append(model)
    
    return models, scaler, device


def load_classification_model():
    """Load classification model."""
    model_dir = parent_dir / 'models' / 'top10_classification'
    model_path = model_dir / 'f1_classifier_top10.pth'
    scaler_path = model_dir / 'scaler_top10_classification.pkl'
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    device = torch.device('cpu')
    input_size = getattr(scaler, 'n_features_in_', 9)
    
    # Check for ensemble models
    ensemble_models = []
    for i in range(3):
        ensemble_path = model_dir / f'f1_classifier_top10_ensemble_{i}.pth'
        if ensemble_path.exists():
            ensemble_models.append(ensemble_path)
    
    if len(ensemble_models) == 3:
        # Load ensemble
        models = []
        for i, path in enumerate(ensemble_models):
            model = F1ClassificationNetwork(
                input_size=input_size,
                hidden_sizes=[256, 128, 64],
                dropout_rate=0.3
            )
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            model.to(device)
            models.append(model)
        print("  Classification ensemble loaded")
        return models, scaler, device
    else:
        # Load single model
        model = F1ClassificationNetwork(
            input_size=input_size,
            hidden_sizes=[256, 128, 64],
            dropout_rate=0.3
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        print("  Classification model loaded")
        return model, scaler, device


def predict_regression(X_scaled, models, device):
    """Predict using regression model (ensemble)."""
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        all_predictions = []
        
        for model in models:
            pred = model(X_tensor).cpu().numpy()
            if pred.ndim > 1:
                pred = pred.flatten()
            all_predictions.append(pred)
        
        # Average predictions
        predictions = np.mean(all_predictions, axis=0)
        
        # Clip to valid range (1-10)
        predictions = np.clip(predictions, 1, 10)
        
        # For regression, just rank by predicted values (lower = better position)
        # Sort by predicted position and assign ranks
        sorted_indices = np.argsort(predictions)
        positions = np.zeros(len(predictions), dtype=int)
        positions[sorted_indices] = np.arange(1, len(predictions) + 1)
        
        return positions


def predict_classification(X_scaled, model, device):
    """Predict using classification model with Hungarian algorithm. Supports ensemble."""
    # Handle ensemble
    if isinstance(model, list):
        # Ensemble: average probabilities from all models
        ensemble_probs = None
        for m in model:
            m.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(device)
                logits = m(X_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                if ensemble_probs is None:
                    ensemble_probs = probs
                else:
                    ensemble_probs += probs
        probs = ensemble_probs / len(model)
    else:
        # Single model
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
    
    n_drivers = len(X_scaled)
    
    if n_drivers == 10:
        # Cost matrix: negative log probabilities
        cost_matrix = -np.log(probs + 1e-10)
        
        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Assign positions (col_indices are position classes 0-9, convert to 1-10)
        positions = np.zeros(n_drivers, dtype=int)
        for i in range(len(row_indices)):
            driver_idx = row_indices[i]
            position_class = col_indices[i]
            positions[driver_idx] = position_class + 1
        
        return positions
    elif n_drivers < 10:
        # Greedy assignment for fewer than 10 drivers
        positions = np.zeros(n_drivers, dtype=int)
        used_positions = set()
        driver_max_probs = np.max(probs, axis=1)
        driver_order = np.argsort(-driver_max_probs)
        
        for driver_idx in driver_order:
            driver_probs = probs[driver_idx]
            best_pos = None
            best_prob = -1
            for pos_class in range(10):
                pos = pos_class + 1
                if pos not in used_positions:
                    if driver_probs[pos_class] > best_prob:
                        best_prob = driver_probs[pos_class]
                        best_pos = pos
            if best_pos is not None:
                positions[driver_idx] = best_pos
                used_positions.add(best_pos)
        
        return positions
    else:
        # More than 10 drivers - select top 10 and use Hungarian
        driver_max_probs = np.max(probs, axis=1)
        top10_indices = np.argsort(-driver_max_probs)[:10]
        top10_probs = probs[top10_indices]
        cost_matrix = -np.log(top10_probs + 1e-10)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        positions = np.zeros(n_drivers, dtype=int)
        for i in range(len(row_indices)):
            driver_idx = top10_indices[row_indices[i]]
            positions[driver_idx] = col_indices[i] + 1
        
        return positions


def evaluate_predictions(y_true, y_pred):
    """Calculate evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    exact_acc = accuracy_score(y_true, y_pred)
    within_1 = np.mean(np.abs(y_true - y_pred) <= 1)
    within_3 = np.mean(np.abs(y_true - y_pred) <= 3)
    
    # Top-3 accuracy: exact position match for drivers who finished in top 3
    top3_mask = y_true <= 3
    if top3_mask.sum() > 0:
        top3_acc = accuracy_score(y_true[top3_mask], y_pred[top3_mask])
    else:
        top3_acc = 0.0
    
    return {
        'mae': mae,
        'exact_accuracy': exact_acc,
        'within_1': within_1,
        'within_3': within_3,
        'top3_accuracy': top3_acc
    }


def main():
    """Compare classification vs regression models."""
    print("=" * 70)
    print("Classification vs Regression Model Comparison")
    print("=" * 70)
    
    # Load data
    print("\nLoading test data...")
    _, test_df, _ = load_data()
    
    # Prepare test data (top 10 only)
    print("Preparing test data (top 10 only)...")
    X_test, y_test, _, _, _ = prepare_features_and_labels(
        test_df,
        filter_dnf=True,
        filter_outliers=True,
        outlier_threshold=6,
        top10_only=True
    )
    
    # Get race groupings for proper evaluation
    test_df_filtered = test_df[
        (test_df['ActualPosition'].between(1, 10)) &
        (~test_df['IsDNF']) &
        (test_df['ActualPosition'].notna())
    ].copy()
    
    # Remove outliers
    if 'GridPosition' in test_df_filtered.columns:
        test_df_filtered = test_df_filtered[
            (test_df_filtered['ActualPosition'] - test_df_filtered['GridPosition'] <= 6) |
            (test_df_filtered['GridPosition'].isna())
        ]
    
    print(f"Test samples: {len(X_test)}")
    print(f"Test races: {test_df_filtered.groupby(['Year', 'RoundNumber']).ngroups}")
    
    # Load models
    print("\nLoading regression model (ensemble)...")
    reg_models, reg_scaler, device = load_regression_model()
    print(f"  Loaded {len(reg_models)} ensemble models")
    
    print("Loading classification model...")
    cls_model, cls_scaler, _ = load_classification_model()
    print("  Classification model loaded")
    
    # Scale features - handle different feature sets
    # Regression and classification may use different features
    # Get the feature columns that each model expects
    reg_feature_cols = getattr(reg_scaler, 'feature_names_in_', None)
    cls_feature_cols = getattr(cls_scaler, 'feature_names_in_', None)
    
    # If scalers don't have feature names, use the feature columns from prepare_features_and_labels
    if reg_feature_cols is None:
        # Regression uses standard features
        reg_feature_cols = ['SeasonPoints', 'SeasonStanding', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                           'ConstructorStanding', 'ConstructorTrackAvg', 'GridPosition', 'RecentForm', 'TrackType']
    if cls_feature_cols is None:
        # Classification may have additional features
        cls_feature_cols = ['SeasonPoints', 'SeasonStanding', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                           'ConstructorStanding', 'ConstructorTrackAvg', 'GridPosition', 'RecentForm', 'TrackType',
                           'PointsGapToLeader', 'FormTrend']
    
    # Prepare features once - both models use same features now
    X_test, y_test, _, _, _ = prepare_features_and_labels(
        test_df_filtered,
        filter_dnf=True,
        filter_outliers=True,
        outlier_threshold=6,
        top10_only=True
    )
    
    # Check feature counts match scalers
    reg_n_features = getattr(reg_scaler, 'n_features_in_', X_test.shape[1])
    cls_n_features = getattr(cls_scaler, 'n_features_in_', X_test.shape[1])
    
    if X_test.shape[1] != reg_n_features:
        print(f"Warning: X_test has {X_test.shape[1]} features but reg_scaler expects {reg_n_features}")
        # Try to select only the features the scaler expects
        if X_test.shape[1] > reg_n_features:
            X_test = X_test[:, :reg_n_features]
    
    if X_test.shape[1] != cls_n_features:
        print(f"Warning: X_test has {X_test.shape[1]} features but cls_scaler expects {cls_n_features}")
        # For classification, if we have fewer features, pad with zeros
        if X_test.shape[1] < cls_n_features:
            padding = np.zeros((X_test.shape[0], cls_n_features - X_test.shape[1]))
            X_test_cls_input = np.hstack([X_test, padding])
        elif X_test.shape[1] > cls_n_features:
            X_test_cls_input = X_test[:, :cls_n_features]
        else:
            X_test_cls_input = X_test
    else:
        X_test_cls_input = X_test
    
    # Scale features
    X_test_reg = reg_scaler.transform(X_test)
    X_test_cls = cls_scaler.transform(X_test_cls_input)
    
    # Handle any NaN or inf values
    X_test_reg = np.nan_to_num(X_test_reg, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_cls = np.nan_to_num(X_test_cls, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Make predictions per race
    print("\nMaking predictions per race...")
    y_pred_reg_all = []
    y_pred_cls_all = []
    y_test_all = []
    
    races = test_df_filtered.groupby(['Year', 'RoundNumber', 'EventName'])
    race_idx_map = {}  # Map race to indices in test_df_filtered
    for idx, (year, round_num, event_name) in enumerate(races.groups.keys()):
        race_df = races.get_group((year, round_num, event_name))
        if len(race_df) < 10:
            continue  # Skip races with fewer than 10 drivers
        race_indices = race_df.index
        race_mask = test_df_filtered.index.isin(race_indices)
        race_idx_map[(year, round_num, event_name)] = race_mask
    
    for (year, round_num, event_name), race_mask in race_idx_map.items():
        race_df = races.get_group((year, round_num, event_name))
        if len(race_df) < 10:
            continue
        
        # Get feature data for this race
        race_X_reg = X_test_reg[race_mask]
        race_X_cls = X_test_cls[race_mask]
        race_y = y_test[race_mask]
        
        # Predict
        race_pred_reg = predict_regression(race_X_reg, reg_models, device)
        race_pred_cls = predict_classification(race_X_cls, cls_model, device)
        
        y_pred_reg_all.extend(race_pred_reg)
        y_pred_cls_all.extend(race_pred_cls)
        y_test_all.extend(race_y)
    
    y_pred_reg = np.array(y_pred_reg_all)
    y_pred_cls = np.array(y_pred_cls_all)
    y_test = np.array(y_test_all)
    
    # Evaluate
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    
    reg_metrics = evaluate_predictions(y_test, y_pred_reg)
    cls_metrics = evaluate_predictions(y_test, y_pred_cls)
    
    print("\nRegression Model (Ensemble):")
    print(f"  MAE:              {reg_metrics['mae']:.4f}")
    print(f"  Exact Accuracy:   {reg_metrics['exact_accuracy']*100:.2f}%")
    print(f"  Within 1 Position: {reg_metrics['within_1']*100:.2f}%")
    print(f"  Within 3 Positions: {reg_metrics['within_3']*100:.2f}%")
    print(f"  Top-3 Accuracy:   {reg_metrics['top3_accuracy']*100:.2f}%")
    
    print("\nClassification Model:")
    print(f"  MAE:              {cls_metrics['mae']:.4f}")
    print(f"  Exact Accuracy:   {cls_metrics['exact_accuracy']*100:.2f}%")
    print(f"  Within 1 Position: {cls_metrics['within_1']*100:.2f}%")
    print(f"  Within 3 Positions: {cls_metrics['within_3']*100:.2f}%")
    print(f"  Top-3 Accuracy:   {cls_metrics['top3_accuracy']*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("DIFFERENCES")
    print("=" * 70)
    
    mae_diff = cls_metrics['mae'] - reg_metrics['mae']
    exact_diff = cls_metrics['exact_accuracy'] - reg_metrics['exact_accuracy']
    within3_diff = cls_metrics['within_3'] - reg_metrics['within_3']
    top3_diff = cls_metrics['top3_accuracy'] - reg_metrics['top3_accuracy']
    
    print(f"\nMAE:              {mae_diff:+.4f} ({'Classification worse' if mae_diff > 0 else 'Classification better'})")
    print(f"Exact Accuracy:   {exact_diff*100:+.2f}% ({'Classification better' if exact_diff > 0 else 'Classification worse'})")
    print(f"Within 3:         {within3_diff*100:+.2f}% ({'Classification better' if within3_diff > 0 else 'Classification worse'})")
    print(f"Top-3 Accuracy:   {top3_diff*100:+.2f}% ({'Classification better' if top3_diff > 0 else 'Classification worse'})")
    
    # Winner
    print("\n" + "=" * 70)
    print("WINNER")
    print("=" * 70)
    
    reg_score = reg_metrics['exact_accuracy'] * 0.5 + (1 - reg_metrics['mae']/10) * 0.3 + reg_metrics['within_3'] * 0.2
    cls_score = cls_metrics['exact_accuracy'] * 0.5 + (1 - cls_metrics['mae']/10) * 0.3 + cls_metrics['within_3'] * 0.2
    
    if cls_score > reg_score:
        print("\nClassification model wins! (Better exact accuracy)")
    elif reg_score > cls_score:
        print("\nRegression model wins! (Better overall balance)")
    else:
        print("\nTie! Both models perform similarly.")
    
    print(f"\nRegression Score: {reg_score:.4f}")
    print(f"Classification Score: {cls_score:.4f}")


if __name__ == '__main__':
    main()

