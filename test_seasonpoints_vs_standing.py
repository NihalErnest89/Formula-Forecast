"""
Test script to compare:
1. Baseline (with SeasonPoints)
2. Without SeasonPoints
3. With SeasonStanding (championship position) instead of SeasonPoints
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
PositionAwareLoss = train_module.PositionAwareLoss


def calculate_season_standing(df: pd.DataFrame) -> pd.Series:
    """
    Calculate driver's championship position (1 = leader, higher = worse) for each race.
    Based on cumulative points up to that race.
    """
    df = df.copy()
    df['SeasonStanding'] = np.nan
    
    # Group by year and process each season
    for year in df['Year'].unique():
        year_data = df[df['Year'] == year].copy()
        
        # Sort by round number
        year_data = year_data.sort_values('RoundNumber')
        
        # For each round, calculate standings based on points up to that round
        for round_num in sorted(year_data['RoundNumber'].unique()):
            # Get all races up to and including this round
            races_up_to_round = year_data[year_data['RoundNumber'] <= round_num]
            
            # Calculate cumulative points per driver
            if 'Points' in races_up_to_round.columns:
                driver_points = races_up_to_round.groupby('DriverNumber')['Points'].sum().sort_values(ascending=False)
                
                # Assign standings (1 = most points)
                standings_dict = {}
                for rank, (driver_num, points) in enumerate(driver_points.items(), 1):
                    standings_dict[driver_num] = rank
                
                # Update standings for this round
                round_mask = (df['Year'] == year) & (df['RoundNumber'] == round_num)
                for idx in df[round_mask].index:
                    driver_num = df.loc[idx, 'DriverNumber']
                    if driver_num in standings_dict:
                        df.at[idx, 'SeasonStanding'] = standings_dict[driver_num]
                    else:
                        # Driver has no points yet - assign worst position
                        df.at[idx, 'SeasonStanding'] = len(standings_dict) + 1
    
    return df['SeasonStanding']


def prepare_features_with_variation(df: pd.DataFrame, variation: str = 'baseline',
                                   filter_dnf=True, filter_outliers=True, top10_only=False):
    """
    Prepare features with different variations:
    - 'baseline': Current features (with SeasonPoints)
    - 'no_seasonpoints': Remove SeasonPoints
    - 'season_standing': Replace SeasonPoints with SeasonStanding
    """
    # Calculate SeasonStanding if needed
    if variation == 'season_standing':
        df = df.copy()
        df['SeasonStanding'] = calculate_season_standing(df)
    
    # Use the standard prepare_features_and_labels but modify feature columns
    X, y, _, _, filter_stats = prepare_features_and_labels(
        df, filter_dnf=filter_dnf, filter_outliers=filter_outliers, top10_only=top10_only
    )
    
    # Get the feature columns that were used
    # We need to modify them based on variation
    if variation == 'no_seasonpoints':
        # Remove SeasonPoints column
        feature_cols = ['SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                       'ConstructorStanding', 'ConstructorTrackAvg', 'GridPosition', 'RecentForm', 'TrackType']
    elif variation == 'season_standing':
        # Replace SeasonPoints with SeasonStanding
        # First, we need to recalculate with SeasonStanding
        # Actually, we need to modify the dataframe before calling prepare_features_and_labels
        # Let me do this differently - modify the dataframe first
        pass
    else:
        # Baseline - use all current features
        feature_cols = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                       'ConstructorStanding', 'ConstructorTrackAvg', 'GridPosition', 'RecentForm', 'TrackType']
    
    # Actually, prepare_features_and_labels uses hardcoded feature_cols
    # We need to modify the dataframe before calling it, or modify the function
    # Let me take a different approach: modify the dataframe columns
    
    return X, y, filter_stats


def prepare_features_custom(df: pd.DataFrame, feature_cols: list,
                            filter_dnf=True, filter_outliers=True, top10_only=False):
    """
    Prepare features with custom feature columns.
    This is a modified version that allows us to specify which features to use.
    """
    df = df.copy()
    
    # Apply filters
    valid_mask = pd.Series([True] * len(df), index=df.index)
    
    if top10_only and 'ActualPosition' in df.columns:
        valid_mask = valid_mask & (df['ActualPosition'] <= 10)
    
    if filter_dnf and 'IsDNF' in df.columns:
        dnf_mask = df['IsDNF'].fillna(False).astype(bool)
        valid_mask = valid_mask & ~dnf_mask
    
    if filter_outliers and 'GridPosition' in df.columns and 'ActualPosition' in df.columns:
        valid_for_outlier_check = valid_mask & df['ActualPosition'].notna() & df['GridPosition'].notna()
        if valid_for_outlier_check.any():
            position_diff = df.loc[valid_for_outlier_check, 'ActualPosition'] - df.loc[valid_for_outlier_check, 'GridPosition']
            outlier_mask_local = position_diff > 6
            outlier_mask = pd.Series(False, index=df.index)
            outlier_mask.loc[valid_for_outlier_check] = outlier_mask_local
            valid_mask = valid_mask & ~outlier_mask
    
    df_filtered = df[valid_mask].copy()
    
    # Ensure all required columns exist
    missing_cols = [col for col in feature_cols if col not in df_filtered.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Select features
    X = df_filtered[feature_cols].copy()
    y = df_filtered['ActualPosition'].values
    
    # Handle missing values
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(median_val)
    
    return X.values, y


def train_model(X_train, y_train, X_val, y_val, epochs=100, 
                learning_rate=0.005, device='cpu', hidden_sizes=[128, 64, 32]):
    """Train a single model."""
    train_dataset = F1Dataset(X_train, y_train)
    val_dataset = F1Dataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = F1NeuralNetwork(input_size=X_train.shape[1], 
                           hidden_sizes=hidden_sizes, 
                           dropout_rate=0.4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=False)
    
    criterion = PositionAwareLoss(exact_weight=5.0, base_loss='huber', delta=1.0)
    
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
    print("Testing SeasonPoints vs SeasonStanding")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_df, test_df, _ = load_data()
    
    # Calculate SeasonStanding for both dataframes
    print("\nCalculating SeasonStanding...")
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['SeasonStanding'] = calculate_season_standing(train_df)
    test_df['SeasonStanding'] = calculate_season_standing(test_df)
    
    # Prepare features for each variation - test all combinations
    variations = [
        ('baseline (SeasonPoints only)', ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                     'ConstructorStanding', 'ConstructorTrackAvg', 'GridPosition', 'RecentForm', 'TrackType']),
        ('SeasonStanding only', ['SeasonStanding', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                            'ConstructorStanding', 'ConstructorTrackAvg', 'GridPosition', 'RecentForm', 'TrackType']),
        ('Both SeasonPoints + SeasonStanding', ['SeasonPoints', 'SeasonStanding', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                            'ConstructorStanding', 'ConstructorTrackAvg', 'GridPosition', 'RecentForm', 'TrackType']),
        ('Neither (removed both)', ['SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                            'ConstructorStanding', 'ConstructorTrackAvg', 'GridPosition', 'RecentForm', 'TrackType']),
    ]
    
    # Time-based split: use 2024 as validation
    train_df_filtered = train_df[
        (train_df['ActualPosition'] <= 10) & 
        (train_df['ActualPosition'].notna()) &
        (~train_df['Status'].str.contains('DNF|DSQ|DNS', na=False))
    ]
    
    if 'GridPosition' in train_df_filtered.columns and 'ActualPosition' in train_df_filtered.columns:
        position_diff = train_df_filtered['ActualPosition'] - train_df_filtered['GridPosition']
        train_df_filtered = train_df_filtered[position_diff <= 6]
    
    val_mask = train_df_filtered['Year'] == 2024
    train_mask = ~val_mask
    
    results = []
    
    for var_name, feature_cols in variations:
        print(f"\n{'=' * 70}")
        print(f"Testing: {var_name}")
        print(f"Features: {', '.join(feature_cols)}")
        print(f"{'=' * 70}")
        
        # Prepare features
        X_train_all, y_train_all = prepare_features_custom(
            train_df, feature_cols, filter_dnf=True, filter_outliers=True, top10_only=True
        )
        X_test, y_test = prepare_features_custom(
            test_df, feature_cols, filter_dnf=True, filter_outliers=True, top10_only=True
        )
        
        # Align with time-based split
        if len(train_df_filtered) == len(X_train_all):
            X_train_split = X_train_all[train_mask]
            y_train_split = y_train_all[train_mask]
            X_val_split = X_train_all[val_mask]
            y_val_split = y_train_all[val_mask]
        else:
            # Fallback
            split_idx = int(len(X_train_all) * 0.8)
            X_train_split = X_train_all[:split_idx]
            y_train_split = y_train_all[:split_idx]
            X_val_split = X_train_all[split_idx:]
            y_val_split = y_train_all[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_split = scaler.fit_transform(X_train_split)
        X_val_split = scaler.transform(X_val_split)
        X_test = scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train_split)}")
        print(f"Validation samples: {len(X_val_split)}")
        print(f"Test samples: {len(X_test)}")
        
        # Train model
        model, metrics = train_model(
            X_train_split, y_train_split,
            X_val_split, y_val_split,
            epochs=150,
            learning_rate=0.005,
            device='cpu'
        )
        
        # Evaluate on test set
        test_dataset = F1Dataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        criterion = PositionAwareLoss(exact_weight=5.0, base_loss='huber', delta=1.0)
        test_loss, test_mae, test_rmse, test_r2, test_exact, test_w1, test_w2, test_w3, _, _ = evaluate_model(
            model, test_loader, criterion, device='cpu'
        )
        
        results.append({
            'variation': var_name,
            'features': len(feature_cols),
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
    print("SUMMARY - Feature Comparison")
    print(f"{'=' * 70}")
    print(f"{'Variation':<20} {'Features':<10} {'Val MAE':<10} {'Val Exact':<12} {'Test MAE':<10} {'Test Exact':<12} {'Test W3':<10}")
    print("-" * 70)
    
    baseline = results[0]
    
    for r in results:
        mae_diff = r['test_mae'] - baseline['test_mae']
        exact_diff = r['test_exact'] - baseline['test_exact']
        w3_diff = r['test_w3'] - baseline['test_w3']
        
        mae_str = f"{r['test_mae']:.3f} ({mae_diff:+.3f})" if abs(mae_diff) > 0.001 else f"{r['test_mae']:.3f}"
        exact_str = f"{r['test_exact']:.1f}% ({exact_diff:+.1f}%)" if abs(exact_diff) > 0.1 else f"{r['test_exact']:.1f}%"
        w3_str = f"{r['test_w3']:.1f}% ({w3_diff:+.1f}%)" if abs(w3_diff) > 0.1 else f"{r['test_w3']:.1f}%"
        
        print(f"{r['variation']:<20} {r['features']:<10} {r['val_mae']:<10.3f} {r['val_exact']:<12.1f} {mae_str:<10} {exact_str:<12} {w3_str:<10}")
    
    print(f"\n{'=' * 70}")
    print("Recommendation:")
    best = min(results, key=lambda x: x['test_mae'])
    if best['variation'] != 'baseline':
        print(f"  Use: {best['variation']}")
        print(f"  Improves MAE by {baseline['test_mae'] - best['test_mae']:.3f}")
        print(f"  Exact accuracy: {best['test_exact']:.1f}% vs {baseline['test_exact']:.1f}% ({best['test_exact'] - baseline['test_exact']:+.1f}%)")
    else:
        print(f"  Keep baseline (SeasonPoints)")
        print(f"  Other variations don't improve performance")


if __name__ == "__main__":
    main()

