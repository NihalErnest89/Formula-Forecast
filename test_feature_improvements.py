"""
Test script to evaluate potential feature improvements.
Testing features that are:
1. Predictable (can be calculated before race)
2. Non-redundant with existing features
3. Capture different aspects of performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
import sys
import importlib.util
import warnings
warnings.filterwarnings('ignore')

# Import from top20/train.py
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))
spec = importlib.util.spec_from_file_location("train_module", parent_dir / "top20" / "train.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

F1NeuralNetwork = train_module.F1NeuralNetwork
train_model = train_module.train_model
PositionAwareLoss = train_module.PositionAwareLoss


def calculate_driver_consistency(df: pd.DataFrame, driver_num: str, current_year: int, current_round: int) -> float:
    """
    Calculate consistency: standard deviation of recent finishes (last 5 races).
    Lower = more consistent/predictable.
    Season-specific.
    """
    driver_races = df[
        (df['DriverNumber'] == driver_num) &
        (df['Year'] == current_year) &
        (df['RoundNumber'] < current_round)
    ].sort_values('RoundNumber', ascending=False).head(5)
    
    if driver_races.empty:
        return np.nan
    
    pos_col = 'ActualPosition' if 'ActualPosition' in driver_races.columns else 'Position'
    if pos_col in driver_races.columns:
        positions = driver_races[pos_col].dropna()
        if len(positions) > 1:
            return positions.std()
        elif len(positions) == 1:
            return 0.0  # Perfect consistency (only one race)
    
    return np.nan


def calculate_constructor_track_avg(df: pd.DataFrame, constructor_standing: int, 
                                     track_name: str, current_year: int, current_round: int) -> float:
    """
    Calculate constructor's average finish at this specific track.
    Uses constructor standing as proxy for constructor identity.
    """
    # Get all races at this track by constructors with same standing
    track_races = df[
        (df['EventName'] == track_name) &
        (df['ConstructorStanding'] == constructor_standing) &
        ((df['Year'] < current_year) | ((df['Year'] == current_year) & (df['RoundNumber'] < current_round)))
    ]
    
    if track_races.empty:
        return np.nan
    
    pos_col = 'ActualPosition' if 'ActualPosition' in track_races.columns else 'Position'
    if pos_col in track_races.columns:
        positions = track_races[pos_col].dropna()
        if len(positions) > 0:
            return positions.mean()
    
    return np.nan


def calculate_recent_trend(df: pd.DataFrame, driver_num: str, current_year: int, current_round: int) -> float:
    """
    Calculate recent trend: slope of last 5 finishes (improving = negative, declining = positive).
    Uses linear regression slope on recent finishes.
    """
    driver_races = df[
        (df['DriverNumber'] == driver_num) &
        (df['Year'] == current_year) &
        (df['RoundNumber'] < current_round)
    ].sort_values('RoundNumber', ascending=True).tail(5)
    
    if len(driver_races) < 2:
        return 0.0  # No trend with < 2 races
    
    pos_col = 'ActualPosition' if 'ActualPosition' in driver_races.columns else 'Position'
    if pos_col in driver_races.columns:
        positions = driver_races[pos_col].dropna().values
        rounds = driver_races['RoundNumber'].values[:len(positions)]
        
        if len(positions) >= 2:
            # Simple linear regression slope
            x = np.arange(len(positions))
            slope = np.polyfit(x, positions, 1)[0]
            return slope  # Negative = improving, Positive = declining
    
    return 0.0


def calculate_championship_position(df: pd.DataFrame, driver_num: str, current_year: int, current_round: int) -> float:
    """
    Calculate driver's position in championship standings (1 = leader, higher = worse).
    Based on season points.
    """
    if current_round == 1:
        return np.nan  # No standings yet
    
    season_races = df[
        (df['Year'] == current_year) &
        (df['RoundNumber'] < current_round)
    ]
    
    if season_races.empty:
        return np.nan
    
    # Calculate points per driver
    driver_points = season_races[season_races['DriverNumber'] == driver_num]
    if driver_points.empty:
        return np.nan
    
    if 'Points' in driver_points.columns:
        total_points = driver_points['Points'].sum()
    else:
        return np.nan
    
    # Get all drivers' points
    all_driver_points = season_races.groupby('DriverNumber')['Points'].sum().sort_values(ascending=False)
    
    # Find position (1 = most points)
    position = (all_driver_points > total_points).sum() + 1
    return float(position)


def calculate_track_type_performance(df: pd.DataFrame, driver_num: str, track_type: int,
                                     current_year: int, current_round: int) -> float:
    """
    Calculate driver's average finish at this track type (street vs permanent).
    Season-specific.
    """
    driver_races = df[
        (df['DriverNumber'] == driver_num) &
        (df['Year'] == current_year) &
        (df['RoundNumber'] < current_round) &
        (df['TrackType'] == track_type)
    ]
    
    if driver_races.empty:
        return np.nan
    
    pos_col = 'ActualPosition' if 'ActualPosition' in driver_races.columns else 'Position'
    if pos_col in driver_races.columns:
        positions = driver_races[pos_col].dropna()
        if len(positions) > 0:
            return positions.mean()
    
    return np.nan


def prepare_features_with_new(df: pd.DataFrame, new_features: list, filter_dnf=True, 
                              filter_outliers=True, outlier_threshold=6, top10_only=False):
    """
    Prepare features including new feature(s).
    """
    base_feature_cols = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                        'ConstructorStanding', 'GridPosition', 'RecentForm', 'TrackType']
    
    df = df.copy()
    df = df.sort_values(['Year', 'RoundNumber']).reset_index(drop=True)
    
    # Calculate new features
    if 'Consistency' in new_features:
        df['Consistency'] = np.nan
        for idx, row in df.iterrows():
            df.at[idx, 'Consistency'] = calculate_driver_consistency(
                df, str(row['DriverNumber']), row['Year'], row['RoundNumber']
            )
    
    if 'ConstructorTrackAvg' in new_features:
        df['ConstructorTrackAvg'] = np.nan
        for idx, row in df.iterrows():
            df.at[idx, 'ConstructorTrackAvg'] = calculate_constructor_track_avg(
                df, row['ConstructorStanding'], row['EventName'], row['Year'], row['RoundNumber']
            )
    
    if 'RecentTrend' in new_features:
        df['RecentTrend'] = np.nan
        for idx, row in df.iterrows():
            df.at[idx, 'RecentTrend'] = calculate_recent_trend(
                df, str(row['DriverNumber']), row['Year'], row['RoundNumber']
            )
    
    if 'ChampionshipPosition' in new_features:
        df['ChampionshipPosition'] = np.nan
        for idx, row in df.iterrows():
            df.at[idx, 'ChampionshipPosition'] = calculate_championship_position(
                df, str(row['DriverNumber']), row['Year'], row['RoundNumber']
            )
    
    if 'TrackTypePerformance' in new_features:
        df['TrackTypePerformance'] = np.nan
        for idx, row in df.iterrows():
            df.at[idx, 'TrackTypePerformance'] = calculate_track_type_performance(
                df, str(row['DriverNumber']), row['TrackType'], row['Year'], row['RoundNumber']
            )
    
    # Filter
    pos_col = 'ActualPosition' if 'ActualPosition' in df.columns else 'Position'
    if filter_dnf:
        df = df[df[pos_col].notna() & (df[pos_col] <= 20)]
    
    if filter_outliers:
        if 'GridPosition' in df.columns and pos_col in df.columns:
            df = df[df[pos_col] - df['GridPosition'] <= outlier_threshold]
    
    if top10_only:
        df = df[df[pos_col] <= 10]
    
    # Prepare features
    feature_cols = base_feature_cols + new_features
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].fillna(0).values
    y = df[pos_col].values
    
    return X, y, df, feature_cols


def main():
    print("=" * 80)
    print("Testing New Feature Ideas")
    print("=" * 80)
    print("\nTesting features that are:")
    print("  1. Predictable (calculable before race)")
    print("  2. Non-redundant with existing features")
    print("  3. Capture different performance aspects")
    print()
    
    # Load data
    data_dir = Path(__file__).parent / 'data'
    training_df = pd.read_csv(data_dir / 'training_data.csv')
    test_df = pd.read_csv(data_dir / 'test_data.csv')
    
    training_df = training_df[training_df['EventName'] != 'Pre-Season Testing']
    test_df = test_df[test_df['EventName'] != 'Pre-Season Testing']
    
    print(f"Training samples: {len(training_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # BASELINE
    print("\n" + "=" * 80)
    print("BASELINE (7 features)")
    print("=" * 80)
    
    baseline_cols = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                     'ConstructorStanding', 'GridPosition', 'RecentForm', 'TrackType']
    
    pos_col = 'ActualPosition' if 'ActualPosition' in training_df.columns else 'Position'
    baseline_train = training_df.copy()
    baseline_train = baseline_train[baseline_train[pos_col].notna() & (baseline_train[pos_col] <= 20)]
    if 'GridPosition' in baseline_train.columns:
        baseline_train = baseline_train[baseline_train[pos_col] - baseline_train['GridPosition'] <= 6]
    baseline_train = baseline_train[baseline_train[pos_col] <= 10]
    
    baseline_test = test_df.copy()
    baseline_test = baseline_test[baseline_test[pos_col].notna() & (baseline_test[pos_col] <= 20)]
    if 'GridPosition' in baseline_test.columns:
        baseline_test = baseline_test[baseline_test[pos_col] - baseline_test['GridPosition'] <= 6]
    baseline_test = baseline_test[baseline_test[pos_col] <= 10]
    
    X_train_base = baseline_train[baseline_cols].fillna(0).values
    y_train_base = baseline_train[pos_col].values
    X_test_base = baseline_test[baseline_cols].fillna(0).values
    y_test_base = baseline_test[pos_col].values
    
    scaler_base = StandardScaler()
    X_train_base_scaled = scaler_base.fit_transform(X_train_base)
    X_test_base_scaled = scaler_base.transform(X_test_base)
    
    print("\nTraining baseline...")
    val_size = len(X_test_base_scaled) // 4
    baseline_model, _ = train_model(
        X_train_base_scaled, y_train_base,
        X_val=X_test_base_scaled[:val_size],
        y_val=y_test_base[:val_size],
        hidden_sizes=[128, 64, 32],
        learning_rate=0.003,
        epochs=100,
        early_stop_patience=20
    )
    
    baseline_model.eval()
    with torch.no_grad():
        predictions = baseline_model(torch.FloatTensor(X_test_base_scaled)).squeeze().numpy()
        baseline_mae = np.mean(np.abs(predictions - y_test_base))
        baseline_exact = np.mean(np.round(predictions) == y_test_base) * 100
        baseline_w1 = np.mean(np.abs(np.round(predictions) - y_test_base) <= 1) * 100
        baseline_w2 = np.mean(np.abs(np.round(predictions) - y_test_base) <= 2) * 100
        baseline_w3 = np.mean(np.abs(np.round(predictions) - y_test_base) <= 3) * 100
    
    print(f"Baseline MAE: {baseline_mae:.3f}, Exact: {baseline_exact:.1f}%, W1: {baseline_w1:.1f}%, W2: {baseline_w2:.1f}%, W3: {baseline_w3:.1f}%")
    
    # Test each new feature individually
    features_to_test = [
        ['Consistency'],
        ['ConstructorTrackAvg'],
        ['RecentTrend'],
        ['ChampionshipPosition'],
        ['TrackTypePerformance']
    ]
    
    results = []
    
    for new_feat in features_to_test:
        print("\n" + "=" * 80)
        print(f"TESTING: {new_feat[0]}")
        print("=" * 80)
        
        try:
            X_train_new, y_train_new, train_df_new, feat_cols_new = prepare_features_with_new(
                training_df, new_feat, filter_dnf=True, filter_outliers=True, 
                outlier_threshold=6, top10_only=True
            )
            X_test_new, y_test_new, test_df_new, _ = prepare_features_with_new(
                test_df, new_feat, filter_dnf=True, filter_outliers=True,
                outlier_threshold=6, top10_only=True
            )
            
            print(f"Training samples: {len(X_train_new)}, Test samples: {len(X_test_new)}")
            print(f"Features: {feat_cols_new}")
            
            # Show feature stats
            if new_feat[0] in train_df_new.columns:
                feat_data = train_df_new[new_feat[0]].dropna()
                if len(feat_data) > 0:
                    print(f"  {new_feat[0]} stats: mean={feat_data.mean():.2f}, std={feat_data.std():.2f}, "
                          f"min={feat_data.min():.2f}, max={feat_data.max():.2f}")
            
            scaler_new = StandardScaler()
            X_train_new_scaled = scaler_new.fit_transform(X_train_new)
            X_test_new_scaled = scaler_new.transform(X_test_new)
            
            print("Training...")
            val_size = len(X_test_new_scaled) // 4
            new_model, _ = train_model(
                X_train_new_scaled, y_train_new,
                X_val=X_test_new_scaled[:val_size],
                y_val=y_test_new[:val_size],
                hidden_sizes=[128, 64, 32],
                learning_rate=0.003,
                epochs=100,
                early_stop_patience=20
            )
            
            new_model.eval()
            with torch.no_grad():
                predictions = new_model(torch.FloatTensor(X_test_new_scaled)).squeeze().numpy()
                mae = np.mean(np.abs(predictions - y_test_new))
                exact = np.mean(np.round(predictions) == y_test_new) * 100
                w1 = np.mean(np.abs(np.round(predictions) - y_test_new) <= 1) * 100
                w2 = np.mean(np.abs(np.round(predictions) - y_test_new) <= 2) * 100
                w3 = np.mean(np.abs(np.round(predictions) - y_test_new) <= 3) * 100
            
            improvement = baseline_mae - mae
            results.append({
                'feature': new_feat[0],
                'mae': mae,
                'exact': exact,
                'w1': w1,
                'w2': w2,
                'w3': w3,
                'improvement': improvement,
                'improvement_pct': (improvement / baseline_mae * 100) if baseline_mae > 0 else 0
            })
            
            print(f"Results: MAE={mae:.3f} (change: {improvement:+.3f}, {improvement/baseline_mae*100:+.1f}%)")
            print(f"  Exact: {exact:.1f}%, W1: {w1:.1f}%, W2: {w2:.1f}%, W3: {w3:.1f}%")
            
        except Exception as e:
            print(f"ERROR testing {new_feat[0]}: {e}")
            results.append({
                'feature': new_feat[0],
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Feature':<25} {'MAE':<10} {'Change':<10} {'% Change':<10} {'Exact %':<10} {'W1 %':<10}")
    print("-" * 80)
    print(f"{'Baseline':<25} {baseline_mae:<10.3f} {'-':<10} {'-':<10} {baseline_exact:<10.1f} {baseline_w1:<10.1f}")
    
    for r in results:
        if 'error' not in r:
            print(f"{r['feature']:<25} {r['mae']:<10.3f} {r['improvement']:+.3f} {r['improvement_pct']:+.1f}% {r['exact']:<10.1f} {r['w1']:<10.1f}")
        else:
            print(f"{r['feature']:<25} ERROR: {r['error']}")
    
    # Best feature
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x['improvement'])
        print(f"\nBest feature: {best['feature']} (improvement: {best['improvement']:.3f}, {best['improvement_pct']:.1f}%)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

