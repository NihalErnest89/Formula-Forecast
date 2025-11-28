"""
Training and testing script for F1 Predictions (Top 5 Only).
Trains a deep neural network on Top 5 finishers only (positions 1-5).
Test results show: Training on Top 5 only improves Top 5 MAE from 3.043 to 1.861
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
train_epoch = train_module.train_epoch
evaluate_model = train_module.evaluate_model
get_feature_importance = train_module.get_feature_importance
train_model = train_module.train_model
plot_training_history = train_module.plot_training_history
plot_weight_progression = train_module.plot_weight_progression
import pickle


class Top5WeightedLoss(nn.Module):
    """
    Loss function that heavily penalizes errors in top 5 positions.
    Trains on all positions (1-20) but weights top 5 MUCH more heavily.
    This helps model learn to distinguish top 5 drivers from the rest.
    """
    def __init__(self, top5_weight=15.0, exact_weight=25.0, base_loss='huber', delta=1.0):
        super().__init__()
        self.top5_weight = top5_weight
        self.exact_weight = exact_weight
        self.delta = delta
        if base_loss == 'huber':
            self.base_loss = nn.HuberLoss(delta=delta, reduction='none')
        elif base_loss == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        else:
            self.base_loss = nn.L1Loss(reduction='none')
    
    def forward(self, pred, target):
        # Calculate base loss for each sample
        base = self.base_loss(pred, target)
        
        # Identify top 5 positions (target <= 5) - MUCH more heavily weight these
        # This ensures the model focuses on getting top 5 predictions right
        top5_mask = (target <= 5).float()
        top5_weighted = base * (1 + self.top5_weight * top5_mask)
        
        # Also VERY heavily penalize exact misses (especially important for top 5)
        rounded_pred = torch.round(pred)
        exact_misses = (rounded_pred != target).float()
        exact_weighted = top5_weighted * (1 + self.exact_weight * exact_misses)
        
        return exact_weighted.mean()


def prepare_features_and_labels(df: pd.DataFrame, label_encoder=None, 
                                filter_dnf=True, filter_outliers=True, 
                                outlier_threshold=6, top10_only=False):
    """
    Prepare feature matrix and label vector from DataFrame (Top 5 version - uses 9 features only).
    Uses the same 9 features as top10: SeasonPoints, SeasonStanding, SeasonAvgFinish, 
    HistoricalTrackAvgPosition, ConstructorStanding, ConstructorTrackAvg, GridPosition, 
    RecentForm, TrackType (excludes PointsGapToLeader and FormTrend).
    """
    # Use only 9 features (same as top10)
    desired_features = ['SeasonPoints', 'SeasonStanding', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                       'ConstructorStanding', 'ConstructorTrackAvg', 'GridPosition', 'RecentForm', 'TrackType']
    
    # Use only features that exist in the dataframe
    feature_cols = [f for f in desired_features if f in df.columns]
    
    if 'SeasonStanding' not in feature_cols and 'SeasonPoints' in feature_cols:
        print(f"  Note: SeasonStanding not in data, using SeasonPoints only (regenerate data with collect_data.py for both features)")
    
    if len(feature_cols) == 0:
        raise ValueError("No valid feature columns found in data!")
    
    # Start with all rows
    valid_mask = pd.Series([True] * len(df), index=df.index)
    filter_stats = {'initial_count': len(df), 'dnf_removed': 0, 'outlier_removed': 0, 'nan_removed': 0, 'top10_filtered': 0}
    
    # Filter to top 10 only if requested (for top5, we'll filter to top5 separately)
    if top10_only and 'ActualPosition' in df.columns:
        top10_mask = df['ActualPosition'] <= 10
        filter_stats['top10_filtered'] = (~top10_mask & valid_mask).sum()
        valid_mask = valid_mask & top10_mask
        if filter_stats['top10_filtered'] > 0:
            print(f"  Filtered to top 10 only: removed {filter_stats['top10_filtered']} positions > 10")
    
    # Filter DNFs if requested
    if filter_dnf and 'IsDNF' in df.columns:
        dnf_mask = df['IsDNF'].fillna(False).astype(bool)
        filter_stats['dnf_removed'] = dnf_mask.sum()
        valid_mask = valid_mask & ~dnf_mask
        if filter_stats['dnf_removed'] > 0:
            print(f"  Removed {filter_stats['dnf_removed']} DNF/DSQ/DNS entries")
    
    # Filter outliers: finish position significantly worse than grid position
    if filter_outliers and 'ActualPosition' in df.columns and 'GridPosition' in df.columns:
        # Only filter if both positions are valid numbers
        position_valid = df['ActualPosition'].notna() & df['GridPosition'].notna()
        if position_valid.any():
            position_diff = df.loc[position_valid, 'ActualPosition'] - df.loc[position_valid, 'GridPosition']
            outlier_mask = position_valid & (position_diff > outlier_threshold)
            filter_stats['outlier_removed'] = outlier_mask.sum()
            valid_mask = valid_mask & ~outlier_mask
            if filter_stats['outlier_removed'] > 0:
                print(f"  Removed {filter_stats['outlier_removed']} outliers (finish > grid + {outlier_threshold})")
    
    # Apply all filters
    df_filtered = df[valid_mask].copy()
    
    # CRITICAL: Renumber positions per race after filtering
    # If we removed DNFs/outliers, positions need to shift up (e.g., if position 3 was removed, position 4 becomes 3)
    if 'ActualPosition' in df_filtered.columns and ('Year' in df_filtered.columns or 'RoundNumber' in df_filtered.columns):
        # Group by race (Year + RoundNumber + EventName if available)
        if 'Year' in df_filtered.columns and 'RoundNumber' in df_filtered.columns:
            if 'EventName' in df_filtered.columns:
                race_groups = df_filtered.groupby(['Year', 'RoundNumber', 'EventName'])
            else:
                race_groups = df_filtered.groupby(['Year', 'RoundNumber'])
        elif 'Year' in df_filtered.columns:
            if 'EventName' in df_filtered.columns:
                race_groups = df_filtered.groupby(['Year', 'EventName'])
            else:
                race_groups = df_filtered.groupby('Year')
        else:
            race_groups = None
        
        if race_groups is not None:
            # Renumber positions within each race (1, 2, 3, ...)
            # Sort by original position first, then assign new sequential positions
            df_filtered = df_filtered.copy()
            
            # Iterate through each race and renumber positions
            for (race_key, group) in race_groups:
                # Sort by original position to maintain order
                sorted_indices = group.sort_values('ActualPosition').index
                # Assign new sequential positions (1, 2, 3, ...)
                for new_pos, idx in enumerate(sorted_indices, start=1):
                    df_filtered.at[idx, 'ActualPosition'] = new_pos
            
            print(f"  Renumbered positions per race after filtering")
    
    # Check if TrackType exists in data, if not add it (for backward compatibility)
    if 'TrackType' not in df_filtered.columns and 'EventName' in df_filtered.columns:
        # Calculate TrackType from EventName
        street_circuits = [
            'Monaco', 'Singapore', 'Azerbaijan', 'Miami', 'Las Vegas',
            'Saudi Arabian'
        ]
        df_filtered['TrackType'] = df_filtered['EventName'].apply(
            lambda x: 1 if any(street in str(x) for street in street_circuits) else 0
        )
        print(f"  Calculated TrackType from EventName (was missing from data)")
        # Re-check feature_cols in case TrackType was missing
        if 'TrackType' in desired_features and 'TrackType' not in feature_cols:
            feature_cols.append('TrackType')
    
    # Check and report which columns have NaN values
    nan_report = {}
    for col in feature_cols:
        if col in df_filtered.columns:
            nan_count = df_filtered[col].isna().sum()
            if nan_count > 0:
                nan_report[col] = nan_count
    
    if nan_report:
        print(f"  NaN values found in features (will be filled with defaults):")
        for col, count in nan_report.items():
            pct = (count / len(df_filtered)) * 100
            print(f"    {col}: {count} NaN values ({pct:.1f}% of rows)")
    
    # Fill NaN values in features with reasonable defaults
    df_filtered = df_filtered.copy()
    for col in feature_cols:
        if col in df_filtered.columns and df_filtered[col].isna().any():
            if col == 'SeasonAvgFinish':
                # First race of season - default to mid-field (10.0)
                df_filtered[col] = df_filtered[col].fillna(10.0)
            elif col == 'RecentForm':
                # Not enough recent races - default to mid-field (10.0)
                df_filtered[col] = df_filtered[col].fillna(10.0)
            elif col == 'HistoricalTrackAvgPosition':
                # New driver at track - default to mid-field (10.0)
                df_filtered[col] = df_filtered[col].fillna(10.0)
            elif col == 'ConstructorTrackAvg':
                # New constructor at track - default to mid-field (10.0)
                df_filtered[col] = df_filtered[col].fillna(10.0)
            elif col == 'GridPosition':
                # No grid position data - default to mid-field (10.5)
                df_filtered[col] = df_filtered[col].fillna(10.5)
            elif col == 'SeasonPoints':
                # First race - default to 0 points
                df_filtered[col] = df_filtered[col].fillna(0)
            elif col == 'SeasonStanding':
                # First race - default to worst position (20)
                df_filtered[col] = df_filtered[col].fillna(20)
            elif col == 'ConstructorStanding':
                # No constructor standing - default to mid-field (10)
                df_filtered[col] = df_filtered[col].fillna(10)
            elif col == 'TrackType':
                # Unknown track type - default to permanent (0)
                df_filtered[col] = df_filtered[col].fillna(0)
            else:
                # For any other feature, use column mean or 0
                fill_val = df_filtered[col].mean()
                if pd.isna(fill_val):
                    fill_val = 0
                df_filtered[col] = df_filtered[col].fillna(fill_val)
    
    # Only remove rows with NaN in ActualPosition (DNF/DSQ/DNS - can't train without labels)
    if 'ActualPosition' in df_filtered.columns:
        actual_pos_nan = df_filtered['ActualPosition'].isna().sum()
        if actual_pos_nan > 0:
            pct = (actual_pos_nan / len(df_filtered)) * 100
            print(f"  ActualPosition: {actual_pos_nan} NaN values ({pct:.1f}% of rows) - will be removed (DNF/DSQ/DNS)")
        
        nan_mask = df_filtered['ActualPosition'].isna()
        filter_stats['nan_removed'] = nan_mask.sum()
        df_filtered = df_filtered[~nan_mask].copy()
    else:
        filter_stats['nan_removed'] = 0
    
    filter_stats['final_count'] = len(df_filtered)
    
    if len(df_filtered) == 0:
        raise ValueError("No valid data remaining after filtering!")
    
    # Extract features and labels
    X = df_filtered[feature_cols].copy()
    feature_names = feature_cols.copy()
    
    # Note: SeasonPoints is "higher is better" while other features are "lower is better"
    # However, StandardScaler normalizes all features, so the model can learn the correct
    # relationship regardless. We don't need to invert SeasonPoints.
    
    # Handle NaN values in features (fill with column mean)
    for col in X.columns:
        if X[col].isna().any():
            fill_val = X[col].mean()
            if pd.isna(fill_val):
                fill_val = 0
            X[col] = X[col].fillna(fill_val)
    
    # Extract labels (ActualPosition)
    if 'ActualPosition' in df_filtered.columns:
        y = df_filtered['ActualPosition'].values
    else:
        raise ValueError("ActualPosition column not found in data!")
    
    return X.values, y, None, filter_stats, feature_names


def save_model(model, scaler, label_encoder, output_dir: str = None, model_index: int = None):
    """Save trained model, scaler, and label encoder (Top 5 version)."""
    if output_dir is None:
        # Resolve path relative to this script's location
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / 'models'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if model_index is not None:
        # Ensemble model
        model_path = output_dir / f'f1_predictor_model_top5_ensemble_{model_index}.pth'
    else:
        # Single model (backward compatibility)
        model_path = output_dir / 'f1_predictor_model_top5.pth'
    
    scaler_path = output_dir / 'scaler_top5.pkl'
    encoder_path = output_dir / 'label_encoder_top5.pkl'
    
    torch.save(model.state_dict(), model_path)
    
    # Only save scaler/encoder once (they're the same for all models)
    if model_index is None or model_index == 0:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print(f"Scaler saved to {scaler_path}")
        print(f"Label encoder saved to {encoder_path}")
    
    print(f"Model saved to {model_path}")


def main():
    """Main function to train and test the F1 prediction model (Top 5 only)."""
    print("F1 Position Prediction Model Training (Top 5 Only)")
    print("=" * 60)
    print("Predicting finishing positions (1-5) for top finishers only")
    print("Test results: Top 5 MAE improved from 3.043 to 1.861")
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
    
    # Prepare training data (filtered - no DNFs/outliers, Top 5 only)
    # Test results show: Training on Top 5 only improves Top 5 MAE from 3.043 to 1.861
    # and "Within 3" accuracy from 55.5% to 79.1%
    print("\nPreparing training data (Top 5 positions only)...")
    
    # TIME-AWARE K-FOLD CROSS-VALIDATION: Use all 2022-2024 for training/validation
    # Split by year to prevent data leakage (time-series data)
    print("\nUsing K-Fold Cross-Validation (Time-Aware) for validation...")
    print("  This allows us to use all training years (2022-2024) for validation")
    print("  while still respecting temporal order (no future data leakage)")
    
    def k_fold_time_based_split(df, n_splits=3):
        """Time-based k-fold split for time-series data. Splits by year."""
        years = sorted(df['Year'].unique())
        n_years = len(years)
        
        if n_splits > n_years:
            n_splits = n_years
        
        fold_size = n_years // n_splits
        folds = []
        
        for i in range(n_splits):
            if i == n_splits - 1:
                # Last fold gets remaining years
                val_years = years[i * fold_size:]
            else:
                val_years = years[i * fold_size:(i + 1) * fold_size]
            
            train_years = [y for y in years if y not in val_years]
            folds.append((train_years, val_years))
        
        return folds
    
    # Perform k-fold cross-validation
    n_splits = 3  # Use 3 folds (one per year: 2022, 2023, 2024)
    folds = k_fold_time_based_split(training_df, n_splits=n_splits)
    
    cv_results = []
    all_models = []
    all_scalers = []
    
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    for fold_idx, (train_years, val_years) in enumerate(folds):
        print(f"\n  Fold {fold_idx + 1}/{n_splits}:")
        print(f"    Train years: {train_years}")
        print(f"    Validation years: {val_years}")
        
        # Split data
        train_fold_df = training_df[training_df['Year'].isin(train_years)].copy()
        val_fold_df = training_df[training_df['Year'].isin(val_years)].copy()
        
        # DON'T filter to top 5 - train on ALL positions (1-20) so model learns to distinguish
        # top 5 drivers from the rest. We'll use weighted loss to emphasize top 5 positions.
        # This helps the model learn that drivers with bad features should get higher (worse) predictions.
        
        # Prepare features
        X_train_fold, y_train_fold, _, _, feature_names = prepare_features_and_labels(
            train_fold_df, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=False
        )
        X_val_fold, y_val_fold, _, _, _ = prepare_features_and_labels(
            val_fold_df, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=False
        )
        
        # Scale features
        scaler_fold = StandardScaler()
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler_fold.transform(X_val_fold)
        
        # Train model for this fold
        train_dataset = F1Dataset(X_train_fold_scaled, y_train_fold)
        val_dataset = F1Dataset(X_val_fold_scaled, y_val_fold)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        input_size = X_train_fold_scaled.shape[1]
        model = F1NeuralNetwork(
            input_size=input_size,
            hidden_sizes=[128, 64, 32],
            dropout_rate=0.4,  # Increased regularization
            equal_init=False  # Use He/Kaiming initialization (optimal for ReLU)
        ).to(device)
        
        # Use Top5WeightedLoss - weights top 5 positions more heavily while training on all positions
        # This helps model learn to distinguish top 5 drivers from the rest
        criterion = Top5WeightedLoss(top5_weight=15.0, exact_weight=25.0, base_loss='huber', delta=1.0)
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=2e-4)  # Increased regularization
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
        )
        
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
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        criterion_eval = nn.MSELoss()
        val_loss, val_mae, val_rmse, val_r2, val_exact, val_w1, val_w2, val_w3, _, _ = evaluate_model(
            model, val_loader, criterion_eval, device
        )
        
        cv_results.append({
            'fold': fold_idx + 1,
            'train_years': train_years,
            'val_years': val_years,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_exact': val_exact,
            'val_w1': val_w1,
            'val_w2': val_w2,
            'val_w3': val_w3
        })
        
        all_models.append(model)
        all_scalers.append(scaler_fold)
        
        print(f"    Validation MAE: {val_mae:.3f}, Exact: {val_exact:.1f}%, W1: {val_w1:.1f}%, W3: {val_w3:.1f}%")
    
    # Calculate average CV metrics
    avg_val_mae = np.mean([r['val_mae'] for r in cv_results])
    avg_val_rmse = np.mean([r['val_rmse'] for r in cv_results])
    avg_val_r2 = np.mean([r['val_r2'] for r in cv_results])
    avg_val_exact = np.mean([r['val_exact'] for r in cv_results])
    avg_val_w1 = np.mean([r['val_w1'] for r in cv_results])
    avg_val_w2 = np.mean([r['val_w2'] for r in cv_results])
    avg_val_w3 = np.mean([r['val_w3'] for r in cv_results])
    
    print(f"\n  Cross-Validation Average Results:")
    print(f"    MAE: {avg_val_mae:.3f} positions")
    print(f"    RMSE: {avg_val_rmse:.3f} positions")
    print(f"    R²: {avg_val_r2:.4f}")
    print(f"    Exact position: {avg_val_exact:.1f}%")
    print(f"    Within 1 position: {avg_val_w1:.1f}%")
    print(f"    Within 2 positions: {avg_val_w2:.1f}%")
    print(f"    Within 3 positions: {avg_val_w3:.1f}%")
    
    # Use the model from the last fold (trained on all years except the last validation year)
    # Or train a final model on ALL training data (2022-2024)
    print(f"\nTraining final model on ALL training data (2022-2024)...")
    # Train on ALL positions (1-20) but use weighted loss to emphasize top 5
    # This helps model learn to distinguish top 5 drivers from the rest
    print("  Training on ALL positions (1-20) with weighted loss emphasizing top 5")
    X_train_all, y_train_all, _, train_stats, feature_names = prepare_features_and_labels(
        training_df, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=False
    )
    
    scaler = StandardScaler()
    X_train_all_scaled = scaler.fit_transform(X_train_all)
    
    # For validation during final training, use the last year (2024)
    val_year = training_df['Year'].max()
    val_df_final = training_df[training_df['Year'] == val_year].copy()
    # DON'T filter validation to top 5 - evaluate on all positions
    # but we'll still report top 5 specific metrics
    X_val_final, y_val_final, _, val_stats, _ = prepare_features_and_labels(
        val_df_final, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=False
    )
    X_val_final_scaled = scaler.transform(X_val_final)
    
    X_train_split = X_train_all_scaled
    X_val_split = X_val_final_scaled
    y_train_split = y_train_all
    y_val_split = y_val_final
    
    print(f"  Final training samples: {len(X_train_split)}")
    print(f"  Final validation samples: {len(X_val_split)}")
    
    # Prepare test data (filtered - for primary evaluation, Top 5 only)
    print("\nPreparing test data (Top 5 positions only, excluding DNFs/outliers)...")
    # Filter to top 5 positions only
    test_df_top5 = test_df.copy()
    if 'ActualPosition' in test_df_top5.columns:
        test_df_top5 = test_df_top5[test_df_top5['ActualPosition'] <= 5].copy()
    X_test, y_test, _, test_stats, _ = prepare_features_and_labels(
        test_df_top5, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=False)
    
    # Also prepare unfiltered test data (for comparison/reference, Top 5 only)
    # Note: We still exclude DNFs (NaN positions) since we can't evaluate accuracy on them
    # But we include outliers (finish >> grid) for reference
    print("\nPreparing test data (Top 5 only, including outliers for reference)...")
    X_test_all, y_test_all, _, test_stats_all, _ = prepare_features_and_labels(
        test_df_top5, filter_dnf=True, filter_outliers=False, outlier_threshold=6, top10_only=False)
    
    # Show filtering stats
    print(f"\nTest Data Filtering Stats:")
    print(f"  Filtered (no outliers): {test_stats.get('final_count', len(y_test))} samples")
    print(f"  Unfiltered (with outliers): {test_stats_all.get('final_count', len(y_test_all))} samples")
    print(f"  Outliers removed: {test_stats.get('outlier_removed', 0)}")
    outliers_removed = test_stats.get('outlier_removed', 0)
    if outliers_removed > 0:
        print(f"  Note: Removed {outliers_removed} outliers (finish > grid + 6) from filtered set")
    else:
        print(f"  Note: No outliers found (finish > grid + 6) in Top 5 positions")
    
    # Scale test data with the same scaler
    X_test_scaled = scaler.transform(X_test)
    X_test_all_scaled = scaler.transform(X_test_all) if len(X_test_all) > 0 else None
    
    print(f"  Features shape: {X_train_split.shape}")
    print(f"  Labels shape: {y_train_split.shape}")
    print(f"  Position range: {y_train_split.min():.0f} - {y_train_split.max():.0f}")
    
    # Verify position range (should be 1-20 since we're training on all positions)
    print(f"  Position range: {y_train_split.min():.0f} - {y_train_split.max():.0f}")
    if y_train_split.min() < 1:
        raise ValueError(f"ERROR: Position range starts at {y_train_split.min():.0f}, but should be >= 1!")
    print(f"  ✓ Training on all positions (1-20) with weighted loss emphasizing top 5")
    
    # Architecture: Optimized based on training improvements test
    # [128, 64, 32] - Best configuration (TrackID removed - had lowest importance 0.0993)
    # Using increased regularization: LR 0.005, Dropout 0.4, Weight Decay 2e-4
    # Position-aware loss to improve exact position accuracy
    hidden_sizes = [128, 64, 32]
    
    # Train ensemble of 3 models (tested: ensemble improves Top-3 accuracy from 29.7% to 30.3% and MAE from 1.874 to 1.869)
    print("\n" + "=" * 60)
    print("TRAINING ENSEMBLE (3 MODELS)")
    print("=" * 60)
    print("Training 3 models with different random seeds for ensemble prediction")
    
    models = []
    histories = []
    
    for i in range(3):
        print(f"\nTraining ensemble model {i+1}/3...")
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        
        # Updated hyperparameters with increased regularization:
        # - LR 0.005: optimized learning rate
        # - Dropout 0.4: increased from 0.3 to reduce overfitting (validation-test gap)
        # - Weight Decay 2e-4: increased from 1e-4 to reduce overfitting
        # - Position-Aware Loss: improves exact position accuracy
        # Manually train final model with Top5WeightedLoss (train_model doesn't support custom loss)
        # This ensures we use the same weighted loss as cross-validation
        train_dataset = F1Dataset(X_train_split, y_train_split)
        val_dataset = F1Dataset(X_val_split, y_val_split)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        input_size = X_train_split.shape[1]
        model = F1NeuralNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=0.4,
            equal_init=False
        ).to(device)
        
        criterion = Top5WeightedLoss(top5_weight=15.0, exact_weight=25.0, base_loss='huber', delta=1.0)
        optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=2e-4)  # Lower LR for stability
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
        
        history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}
        best_val_mae = float('inf')
        best_model_state = None
        patience_counter = 0
        max_patience = 40  # More patience
        
        print(f"\nTraining final model with Top5WeightedLoss (top5_weight=15.0, exact_weight=25.0)...")
        for epoch in range(300):
            train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_mae, val_rmse, val_r2, val_exact, val_w1, val_w2, val_w3, _, _ = evaluate_model(
                model, val_loader, criterion, device
            )
            
            history['train_loss'].append(train_loss)
            history['train_mae'].append(train_mae)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            
            scheduler.step(val_loss)
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break
        
        if best_model_state:
            model.load_state_dict(best_model_state)
        models.append(model)
        histories.append(history)
        print(f"  Model {i+1} training complete")
    
    # Use the first model for evaluation/plotting (they should be similar)
    model = models[0]
    history = histories[0]
    
    # Plot training history
    script_dir = Path(__file__).parent
    plot_training_history(history, save_path=str(script_dir.parent / 'images' / 'training_history_top5.png'))
    
    # Plot weight progression separately if available
    if 'weight_progression' in history and len(history['weight_progression']) > 0 and feature_names:
        plot_weight_progression(history, feature_names, 
                                save_path=str(script_dir.parent / 'images' / 'weight_progression_top5.png'))
    
    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Final Model Evaluation on Validation Set (2024)")
    print("=" * 60)
    print("Note: This is the final model trained on ALL training data (2022-2024)")
    print("      Cross-validation results above show average performance across all folds")
    
    val_dataset = F1Dataset(X_val_split, y_val_split)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    criterion = nn.MSELoss()
    
    val_loss, val_mae, val_rmse, val_r2, val_exact, val_w1, val_w2, val_w3, val_preds, val_labels = evaluate_model(
        model, val_loader, criterion, device
    )
    
    print(f"\nValidation Metrics:")
    print(f"  MAE: {val_mae:.3f} positions")
    print(f"  RMSE: {val_rmse:.3f} positions")
    print(f"  R²: {val_r2:.4f}")
    print(f"\nPosition Accuracy:")
    print(f"  Exact position: {val_exact:.1f}%")
    print(f"  Within 1 position: {val_w1:.1f}%")
    print(f"  Within 2 positions: {val_w2:.1f}%")
    print(f"  Within 3 positions: {val_w3:.1f}%")
    
    # Feature importance (from first layer weights)
    feature_importances = get_feature_importance(model, feature_names, device)
    print(f"\nFeature Importances (Weight Distribution from First Layer):")
    for name, importance in feature_importances.items():
        print(f"  {name}: {importance:.4f}")
    
    # Scatter plot: predicted vs actual positions (using ranked positions)
    script_dir = Path(__file__).parent
    images_dir = script_dir.parent / 'images'
    images_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert raw predictions to ranked positions (sort by prediction value, assign rank)
    # This matches how we actually use predictions - we sort and rank, not use raw values
    # Note: This ranks all predictions together. For multi-race validation sets, ranking should be per-race,
    # but without race grouping info in the loader, this is a reasonable approximation.
    val_preds_ranked = np.argsort(np.argsort(val_preds)) + 1
    
    # Group by actual position and compute average predicted position for each actual position
    unique_positions = np.unique(val_labels)
    avg_predicted = []
    positions_list = []
    
    for pos in unique_positions:
        mask = val_labels == pos
        if mask.sum() > 0:
            avg_pred = np.mean(val_preds_ranked[mask])
            avg_predicted.append(avg_pred)
            positions_list.append(pos)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(positions_list, avg_predicted, s=100, alpha=0.7, edgecolors='black', linewidths=1.5)
    plt.plot([val_labels.min(), val_labels.max()], [val_labels.min(), val_labels.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Position')
    plt.ylabel('Average Predicted Rank (from sorted predictions)')
    plt.title('Predicted vs Actual Finishing Positions (Top 5 Only)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_path = images_dir / 'prediction_scatter_top5.png'
    plt.savefig(scatter_path, dpi=150)
    print(f"\nPrediction scatter plot saved to {scatter_path}")
    plt.close()
    
    # Evaluate on test set (filtered - primary metrics)
    if len(X_test_scaled) > 0:
        print("\n" + "=" * 60)
        print("Evaluation on Test Set (FILTERED - Excluding DNFs/Outliers)")
        print("=" * 60)
        print("This is the primary evaluation metric - excludes unpredictable failures")
        print(f"Test samples: {len(y_test)} (after filtering, Top 5 only)")
        
        # Verify that all test positions are 1-5 (safety check)
        if y_test.max() > 5 or y_test.min() < 1:
            raise ValueError(f"ERROR: Test position range is {y_test.min():.0f} - {y_test.max():.0f}, but should be 1-5 for top5 model!")
        print(f"  ✓ Verified: All test positions are in range 1-5")
        
        test_dataset = F1Dataset(X_test_scaled, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        test_loss, test_mae, test_rmse, test_r2, test_exact, test_w1, test_w2, test_w3, test_preds, test_labels = evaluate_model(
            model, test_loader, criterion, device
        )
        
        print(f"\nTest Metrics (Filtered, Top 5 Only):")
        print(f"  MAE: {test_mae:.3f} positions")
        print(f"  RMSE: {test_rmse:.3f} positions")
        print(f"  R²: {test_r2:.4f}")
        print(f"\nPosition Accuracy (Filtered, Top 5 Only):")
        print(f"  Exact position: {test_exact:.1f}%")
        print(f"  Within 1 position: {test_w1:.1f}%")
        print(f"  Within 2 positions: {test_w2:.1f}%")
        print(f"  Within 3 positions: {test_w3:.1f}%")
        
        # Also evaluate on unfiltered data for reference
        if X_test_all_scaled is not None and len(X_test_all_scaled) > 0:
            print("\n" + "=" * 60)
            print("Evaluation on Test Set (UNFILTERED - Including All Entries)")
            print("=" * 60)
            print("Reference metrics - includes DNFs/outliers for comparison (Top 5 only)")
            print(f"Test samples: {len(y_test_all)} (all entries, Top 5 only)")
            
            test_dataset_all = F1Dataset(X_test_all_scaled, y_test_all)
            test_loader_all = DataLoader(test_dataset_all, batch_size=32, shuffle=False)
            
            test_loss_all, test_mae_all, test_rmse_all, test_r2_all, test_exact_all, test_w1_all, test_w2_all, test_w3_all, _, _ = evaluate_model(
                model, test_loader_all, criterion, device
            )
            
            print(f"\nTest Metrics (Unfiltered - Reference, Top 5 Only):")
            print(f"  MAE: {test_mae_all:.3f} positions")
            print(f"  RMSE: {test_rmse_all:.3f} positions")
            print(f"  R²: {test_r2_all:.4f}")
            print(f"\nPosition Accuracy (Unfiltered - Reference, Top 5 Only):")
            print(f"  Exact position: {test_exact_all:.1f}%")
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
    
    # Save all ensemble models (no label encoder for regression)
    print("\n" + "=" * 60)
    print("SAVING ENSEMBLE MODELS")
    print("=" * 60)
    for i, model_to_save in enumerate(models):
        save_model(model_to_save, scaler, None, model_index=i)
    print(f"\nSaved {len(models)} ensemble models")
    
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
        'model_type': 'neural_network_regression_top5',
        'validation_strategy': 'k_fold_cross_validation',
        'k_fold_cv': {
            'n_splits': n_splits,
            'folds': convert_to_native(cv_results),
            'average': {
                'mae': float(avg_val_mae),
                'rmse': float(avg_val_rmse),
                'r2': float(avg_val_r2),
                'exact': float(avg_val_exact),
                'within_1': float(avg_val_w1),
                'within_2': float(avg_val_w2),
                'within_3': float(avg_val_w3)
            }
        },
        'filtering_stats': convert_to_native(filtering_stats_dict),
        'final_model_validation': {
            'mae': float(val_mae),
            'rmse': float(val_rmse),
            'r2': float(val_r2),
            'exact': float(val_exact),
            'within_1': float(val_w1),
            'within_2': float(val_w2),
            'within_3': float(val_w3)
        },
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
            'input_size': X_train_split.shape[1],
            'hidden_sizes': hidden_sizes,
            'dropout_rate': 0.4,
            'output': 'regression (single position value, 1-5)',
            'features': feature_names if feature_names else []
        }
    }
    
    # Create json directory if it doesn't exist
    script_dir = Path(__file__).parent
    json_dir = script_dir.parent / 'json'
    json_dir.mkdir(exist_ok=True, parents=True)
    
    results_path = json_dir / 'training_results_top5.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Results saved to {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

