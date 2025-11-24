"""
Test script to compare baseline model vs model with RaceGap feature.
RaceGap = Historical average of (FinishPosition - GridPosition)
- Negative values = driver typically loses positions (qualifies better than they finish)
- Positive values = driver typically gains positions (better race pace than qualifying)
- Near zero = driver finishes close to grid position (consistent)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Import from top20/train.py
import sys
from pathlib import Path
import importlib.util
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))
# Import with explicit module name to avoid circular import
spec = importlib.util.spec_from_file_location("train_module", parent_dir / "top20" / "train.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

F1NeuralNetwork = train_module.F1NeuralNetwork
train_model = train_module.train_model
evaluate_model = train_module.evaluate_model
PositionAwareLoss = train_module.PositionAwareLoss

def calculate_race_gap(df: pd.DataFrame, driver_num: str, current_year: int, current_round: int) -> float:
    """
    Calculate average race gap: (FinishPosition - GridPosition) for a driver.
    Season-specific: only uses races from current season before current round.
    
    Args:
        df: DataFrame with all race data
        driver_num: Driver number as string
        current_year: Current race year
        current_round: Current race round number
        
    Returns:
        Average race gap (negative = loses positions, positive = gains positions)
    """
    # Only use races from current season (before current round)
    driver_races = df[
        (df['DriverNumber'] == driver_num) &
        (df['Year'] == current_year) &
        (df['RoundNumber'] < current_round)
    ]
    
    if driver_races.empty:
        return np.nan
    
    # Calculate gap: FinishPosition - ActualGridPosition
    # Note: GridPosition in the data is AvgGridPosition (average), not actual grid position
    # We need to use the actual grid position from the race, which might be in a different column
    pos_col = 'ActualPosition' if 'ActualPosition' in driver_races.columns else 'Position'
    
    # Try to find actual grid position column (might be StartingGrid, Grid, or stored elsewhere)
    grid_col = None
    for col in ['StartingGrid', 'Grid', 'ActualGrid', 'GridPosition']:
        if col in driver_races.columns:
            grid_col = col
            break
    
    if pos_col in driver_races.columns and grid_col:
        # For GridPosition, we need to check if it's the actual grid or average
        # If it's the average (which it is in our data), we can't use it for gap calculation
        # Instead, we'll use a workaround: calculate gap from finish position relative to season average grid
        if grid_col == 'GridPosition':
            # GridPosition is AvgGridPosition, so we calculate: Finish - AvgGrid
            # This gives us how much the driver deviates from their typical qualifying performance
            gaps = driver_races[pos_col] - driver_races[grid_col]
        else:
            # Use actual grid position if available
            gaps = driver_races[pos_col] - driver_races[grid_col]
        
        valid_gaps = gaps.dropna()
        if len(valid_gaps) > 0:
            return valid_gaps.mean()
    
    return np.nan


def prepare_features_with_race_gap(df: pd.DataFrame, filter_dnf=True, filter_outliers=True, 
                                   outlier_threshold=6, top10_only=False):
    """
    Prepare features including the new RaceGap feature.
    """
    # Base features
    feature_cols = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                   'ConstructorStanding', 'GridPosition', 'RecentForm', 'TrackType']
    
    # Add RaceGap feature
    df = df.copy()
    df['RaceGap'] = np.nan
    
    # Calculate RaceGap for each row
    # Sort by year and round to ensure we process in chronological order
    df = df.sort_values(['Year', 'RoundNumber']).reset_index(drop=True)
    
    for idx, row in df.iterrows():
        driver_num = str(row['DriverNumber'])
        year = row['Year']
        round_num = row['RoundNumber']
        
        # Calculate race gap using all data up to this point (before current race)
        race_gap = calculate_race_gap(df, driver_num, year, round_num)
        
        # If no historical data (e.g., first race of season), use NaN instead of 0
        # This will be handled during feature preparation
        df.at[idx, 'RaceGap'] = race_gap
    
    # Filter DNFs and outliers
    pos_col = 'ActualPosition' if 'ActualPosition' in df.columns else 'Position'
    if filter_dnf:
        df = df[df[pos_col].notna() & (df[pos_col] <= 20)]
    
    if filter_outliers:
        # Remove races where finish > grid + threshold
        if 'GridPosition' in df.columns and pos_col in df.columns:
            df = df[df[pos_col] - df['GridPosition'] <= outlier_threshold]
    
    if top10_only:
        df = df[df[pos_col] <= 10]
    
    # Prepare features
    feature_cols_with_gap = feature_cols + ['RaceGap']
    
    # Check if all features exist
    missing_features = [col for col in feature_cols_with_gap if col not in df.columns]
    if missing_features:
        print(f"Warning: Missing features {missing_features}")
        # Use only available features
        feature_cols_with_gap = [col for col in feature_cols_with_gap if col in df.columns]
    
    # Fill NaN RaceGap with 0 (neutral) - this happens for first race of season
    # But first, let's see if we have any non-zero RaceGap values
    if 'RaceGap' in df.columns:
        non_zero_gap = (df['RaceGap'].notna() & (df['RaceGap'] != 0)).sum()
        print(f"  RaceGap: {non_zero_gap} non-zero values out of {len(df)} total")
    
    X = df[feature_cols_with_gap].fillna(0).values
    y = df[pos_col].values
    
    return X, y, df, feature_cols_with_gap


def main():
    print("=" * 80)
    print("Testing RaceGap Feature: Qualifying-to-Race Performance Gap")
    print("=" * 80)
    print("\nRaceGap = Average(FinishPosition - GridPosition)")
    print("  Negative = typically loses positions (qualifies better than finishes)")
    print("  Positive = typically gains positions (better race pace)")
    print("  Near zero = consistent (finishes close to grid)")
    print()
    
    # Load data
    print("Loading data...")
    data_dir = Path(__file__).parent / 'data'
    training_path = data_dir / 'training_data.csv'
    test_path = data_dir / 'test_data.csv'
    
    if not training_path.exists():
        raise FileNotFoundError(f"Training data not found at {training_path}. Run collect_data.py first.")
    
    training_df = pd.read_csv(training_path)
    test_df = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
    
    if test_df.empty:
        raise FileNotFoundError(f"Test data not found at {test_path}. Run collect_data.py first.")
    
    # Filter Pre-Season Testing from test data only (as per user request)
    training_df = training_df[training_df['EventName'] != 'Pre-Season Testing']
    test_df = test_df[test_df['EventName'] != 'Pre-Season Testing']
    
    print(f"Training samples: {len(training_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # BASELINE: 7 features (no RaceGap)
    print("\n" + "=" * 80)
    print("BASELINE MODEL (7 features)")
    print("=" * 80)
    
    baseline_feature_cols = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                            'ConstructorStanding', 'GridPosition', 'RecentForm', 'TrackType']
    
    # Filter and prepare baseline features
    baseline_train = training_df.copy()
    # Use ActualPosition (not Position)
    pos_col = 'ActualPosition' if 'ActualPosition' in baseline_train.columns else 'Position'
    baseline_train = baseline_train[baseline_train[pos_col].notna() & (baseline_train[pos_col] <= 20)]
    if 'GridPosition' in baseline_train.columns and pos_col in baseline_train.columns:
        baseline_train = baseline_train[baseline_train[pos_col] - baseline_train['GridPosition'] <= 6]
    baseline_train = baseline_train[baseline_train[pos_col] <= 10]
    
    baseline_test = test_df.copy()
    baseline_test = baseline_test[baseline_test[pos_col].notna() & (baseline_test[pos_col] <= 20)]
    if 'GridPosition' in baseline_test.columns and pos_col in baseline_test.columns:
        baseline_test = baseline_test[baseline_test[pos_col] - baseline_test['GridPosition'] <= 6]
    baseline_test = baseline_test[baseline_test[pos_col] <= 10]
    
    X_train_base = baseline_train[baseline_feature_cols].fillna(0).values
    y_train_base = baseline_train[pos_col].values
    X_test_base = baseline_test[baseline_feature_cols].fillna(0).values
    y_test_base = baseline_test[pos_col].values
    
    # Scale features
    scaler_base = StandardScaler()
    X_train_base_scaled = scaler_base.fit_transform(X_train_base)
    X_test_base_scaled = scaler_base.transform(X_test_base)
    
    # Train baseline model
    print("\nTraining baseline model...")
    val_size = len(X_test_base_scaled) // 4
    baseline_model, baseline_train_metrics = train_model(
        X_train_base_scaled, y_train_base,
        X_val=X_test_base_scaled[:val_size],
        y_val=y_test_base[:val_size],
        hidden_sizes=[128, 64, 32],
        learning_rate=0.003,
        epochs=200,
        early_stop_patience=30
    )
    
    # Evaluate baseline
    baseline_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_base_scaled)
        y_test_tensor = torch.FloatTensor(y_test_base)
        predictions = baseline_model(X_test_tensor).squeeze().numpy()
        
        mae = np.mean(np.abs(predictions - y_test_base))
        rmse = np.sqrt(np.mean((predictions - y_test_base) ** 2))
        exact = np.mean(np.round(predictions) == y_test_base) * 100
        w1 = np.mean(np.abs(np.round(predictions) - y_test_base) <= 1) * 100
        w2 = np.mean(np.abs(np.round(predictions) - y_test_base) <= 2) * 100
        w3 = np.mean(np.abs(np.round(predictions) - y_test_base) <= 3) * 100
    
    print(f"\nBaseline Test Results:")
    print(f"  MAE: {mae:.3f} positions")
    print(f"  RMSE: {rmse:.3f} positions")
    print(f"  Exact: {exact:.1f}%")
    print(f"  Within 1: {w1:.1f}%")
    print(f"  Within 2: {w2:.1f}%")
    print(f"  Within 3: {w3:.1f}%")
    
    baseline_mae = mae
    baseline_exact = exact
    baseline_w1 = w1
    baseline_w2 = w2
    baseline_w3 = w3
    
    # MODEL WITH RACEGAP: 8 features
    print("\n" + "=" * 80)
    print("MODEL WITH RACEGAP (8 features)")
    print("=" * 80)
    
    # Prepare features with RaceGap
    X_train_gap, y_train_gap, train_df_gap, feature_cols_gap = prepare_features_with_race_gap(
        training_df, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=True
    )
    X_test_gap, y_test_gap, test_df_gap, _ = prepare_features_with_race_gap(
        test_df, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=True
    )
    
    print(f"\nTraining samples (with RaceGap): {len(X_train_gap)}")
    print(f"Test samples (with RaceGap): {len(X_test_gap)}")
    print(f"Features: {feature_cols_gap}")
    
    # Show some RaceGap statistics
    if 'RaceGap' in train_df_gap.columns:
        print(f"\nRaceGap Statistics (Training):")
        print(f"  Mean: {train_df_gap['RaceGap'].mean():.2f}")
        print(f"  Std: {train_df_gap['RaceGap'].std():.2f}")
        print(f"  Min: {train_df_gap['RaceGap'].min():.2f}")
        print(f"  Max: {train_df_gap['RaceGap'].max():.2f}")
        print(f"  Negative (loses positions): {(train_df_gap['RaceGap'] < 0).sum()} samples")
        print(f"  Positive (gains positions): {(train_df_gap['RaceGap'] > 0).sum()} samples")
        print(f"  Neutral (near zero): {(np.abs(train_df_gap['RaceGap']) < 0.5).sum()} samples")
    
    # Scale features
    scaler_gap = StandardScaler()
    X_train_gap_scaled = scaler_gap.fit_transform(X_train_gap)
    X_test_gap_scaled = scaler_gap.transform(X_test_gap)
    
    # Train model with RaceGap
    print("\nTraining model with RaceGap...")
    val_size = len(X_test_gap_scaled) // 4
    gap_model, gap_train_metrics = train_model(
        X_train_gap_scaled, y_train_gap,
        X_val=X_test_gap_scaled[:val_size],
        y_val=y_test_gap[:val_size],
        hidden_sizes=[128, 64, 32],
        learning_rate=0.003,
        epochs=200,
        early_stop_patience=30
    )
    
    # Evaluate model with RaceGap
    gap_model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_gap_scaled)
        y_test_tensor = torch.FloatTensor(y_test_gap)
        predictions = gap_model(X_test_tensor).squeeze().numpy()
        
        mae = np.mean(np.abs(predictions - y_test_gap))
        rmse = np.sqrt(np.mean((predictions - y_test_gap) ** 2))
        exact = np.mean(np.round(predictions) == y_test_gap) * 100
        w1 = np.mean(np.abs(np.round(predictions) - y_test_gap) <= 1) * 100
        w2 = np.mean(np.abs(np.round(predictions) - y_test_gap) <= 2) * 100
        w3 = np.mean(np.abs(np.round(predictions) - y_test_gap) <= 3) * 100
    
    print(f"\nRaceGap Model Test Results:")
    print(f"  MAE: {mae:.3f} positions")
    print(f"  RMSE: {rmse:.3f} positions")
    print(f"  Exact: {exact:.1f}%")
    print(f"  Within 1: {w1:.1f}%")
    print(f"  Within 2: {w2:.1f}%")
    print(f"  Within 3: {w3:.1f}%")
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<20} {'Baseline':<15} {'With RaceGap':<15} {'Change':<15}")
    print("-" * 80)
    print(f"{'MAE':<20} {baseline_mae:<15.3f} {mae:<15.3f} {mae - baseline_mae:+.3f}")
    print(f"{'Exact %':<20} {baseline_exact:<15.1f} {exact:<15.1f} {exact - baseline_exact:+.1f}%")
    print(f"{'Within 1 %':<20} {baseline_w1:<15.1f} {w1:<15.1f} {w1 - baseline_w1:+.1f}%")
    print(f"{'Within 2 %':<20} {baseline_w2:<15.1f} {w2:<15.1f} {w2 - baseline_w2:+.1f}%")
    print(f"{'Within 3 %':<20} {baseline_w3:<15.1f} {w3:<15.1f} {w3 - baseline_w3:+.1f}%")
    
    improvement = baseline_mae - mae
    if improvement > 0:
        print(f"\n[+] RaceGap feature improves MAE by {improvement:.3f} positions ({improvement/baseline_mae*100:.1f}% improvement)")
    else:
        print(f"\n[-] RaceGap feature degrades MAE by {abs(improvement):.3f} positions")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

