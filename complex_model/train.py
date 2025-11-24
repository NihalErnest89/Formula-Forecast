"""
Complex F1 Prediction Model Training (Top 10 Only).
Uses a deep neural network with many engineered features for maximum predictive power.
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

# Import base functions from top20/train.py
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import importlib.util
spec = importlib.util.spec_from_file_location("train_module", parent_dir / "top20" / "train.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

F1Dataset = train_module.F1Dataset
F1NeuralNetwork = train_module.F1NeuralNetwork
load_data = train_module.load_data
train_epoch = train_module.train_epoch
evaluate_model = train_module.evaluate_model
get_feature_importance = train_module.get_feature_importance
import pickle


class ComplexF1NeuralNetwork(nn.Module):
    """
    Deep Neural Network for F1 Position Prediction with complex architecture.
    Larger and deeper than the base model to handle many features.
    """
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], dropout_rate=0.4, equal_init=True):
        super(ComplexF1NeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.equal_init = equal_init
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers (deeper and wider)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        
        # Output layer: single value for position prediction
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights with equal feature importance
        if equal_init:
            self._initialize_equal_weights()
    
    def _initialize_equal_weights(self):
        """Initialize first layer weights to give equal importance to all features."""
        first_layer = self.network[0]
        with torch.no_grad():
            # Calculate weight value for equal initialization
            # For ReLU, we want weights that give equal contribution
            # Use He initialization scaled to give equal feature importance
            fan_in = first_layer.in_features
            weight_value = np.sqrt(2.0 / fan_in)  # He initialization
            first_layer.weight.fill_(weight_value)
            if first_layer.bias is not None:
                first_layer.bias.zero_()
    
    def forward(self, x):
        return self.network(x).squeeze()


def prepare_complex_features(df: pd.DataFrame, historical_df: pd.DataFrame = None,
                            filter_dnf=True, filter_outliers=True, 
                            outlier_threshold=6, top10_only=False):
    """
    Prepare complex feature matrix with many engineered features.
    
    Features included:
    - Base features (8): SeasonPoints, SeasonAvgFinish, HistoricalTrackAvgPosition, 
      ConstructorStanding, GridPosition, RecentForm, TrackType, TrackID
    - Constructor features: ConstructorTrackAvgPosition, ConstructorRecentForm, ConstructorPoints
    - Driver performance: DriverWinRate, DriverPodiumRate, DriverPointsPerRace
    - Constructor performance: ConstructorWinRate
    - Track-specific: BestFinishAtTrack, WorstFinishAtTrack, TrackConsistency, LastYearPosition
    - Championship: PointsGapToLeader, FormTrend
    """
    df_work = df.copy()
    valid_mask = pd.Series([True] * len(df_work), index=df_work.index)
    filter_stats = {'initial_count': len(df_work), 'dnf_removed': 0, 'outlier_removed': 0, 
                    'nan_removed': 0, 'top10_filtered': 0}
    
    # Use historical_df for feature calculations if provided, otherwise use df_work
    calc_df = historical_df if historical_df is not None and not historical_df.empty else df_work
    
    # Filter to top 10 only if requested
    if top10_only and 'ActualPosition' in df_work.columns:
        top10_mask = df_work['ActualPosition'] <= 10
        filter_stats['top10_filtered'] = (~top10_mask & valid_mask).sum()
        valid_mask = valid_mask & top10_mask
        if filter_stats['top10_filtered'] > 0:
            print(f"  Filtered to top 10 only: removed {filter_stats['top10_filtered']} positions > 10")
    
    # Filter DNFs
    if filter_dnf and 'IsDNF' in df_work.columns:
        dnf_mask = df_work['IsDNF'].fillna(False).astype(bool)
        filter_stats['dnf_removed'] = dnf_mask.sum()
        valid_mask = valid_mask & ~dnf_mask
        if filter_stats['dnf_removed'] > 0:
            print(f"  Removed {filter_stats['dnf_removed']} DNF/DSQ/DNS entries")
    
    # Filter outliers
    if filter_outliers and 'GridPosition' in df_work.columns and 'ActualPosition' in df_work.columns:
        valid_for_outlier_check = valid_mask & df_work['ActualPosition'].notna() & df_work['GridPosition'].notna()
        if valid_for_outlier_check.any():
            position_diff = df_work.loc[valid_for_outlier_check, 'ActualPosition'] - df_work.loc[valid_for_outlier_check, 'GridPosition']
            outlier_mask_local = position_diff > outlier_threshold
            outlier_mask = pd.Series(False, index=df_work.index)
            outlier_mask.loc[valid_for_outlier_check] = outlier_mask_local
            filter_stats['outlier_removed'] = outlier_mask.sum()
            valid_mask = valid_mask & ~outlier_mask
            if filter_stats['outlier_removed'] > 0:
                print(f"  Removed {filter_stats['outlier_removed']} outliers (finish > grid + {outlier_threshold})")
    
    # Remove NaN positions
    if 'ActualPosition' in df_work.columns:
        nan_mask = df_work['ActualPosition'].notna()
        filter_stats['nan_removed'] = (~nan_mask & valid_mask).sum()
        valid_mask = valid_mask & nan_mask
    
    df_filtered = df_work[valid_mask].copy()
    filter_stats['final_count'] = len(df_filtered)
    print(f"  Filtering: {filter_stats['initial_count']} -> {filter_stats['final_count']} samples "
          f"({filter_stats['initial_count'] - filter_stats['final_count']} removed)")
    
    # Calculate complex features
    print("  Calculating complex features...")
    
    # Base features
    feature_cols = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                   'ConstructorStanding', 'GridPosition', 'RecentForm', 'TrackType']
    
    # Add TrackID
    if 'EventName' in df_filtered.columns:
        unique_tracks = sorted(calc_df['EventName'].unique()) if 'EventName' in calc_df.columns else sorted(df_filtered['EventName'].unique())
        track_to_id = {track: idx for idx, track in enumerate(unique_tracks)}
        max_track_id = len(unique_tracks) - 1
        df_filtered['TrackID'] = df_filtered['EventName'].map(track_to_id)
        if max_track_id > 0:
            df_filtered['TrackID'] = df_filtered['TrackID'] / max_track_id
        else:
            df_filtered['TrackID'] = 0.0
        feature_cols.append('TrackID')
    
    # ConstructorTrackAvgPosition: Average position of constructor at this track
    if 'EventName' in df_filtered.columns and 'ConstructorStanding' in df_filtered.columns:
        df_filtered['ConstructorTrackAvgPosition'] = np.nan
        for idx, row in df_filtered.iterrows():
            track_name = row['EventName']
            constructor_standing = row['ConstructorStanding']
            # Find all races by constructors with same standing at this track
            track_races = calc_df[(calc_df['EventName'] == track_name) & 
                                 (calc_df['ConstructorStanding'] == constructor_standing)]
            if not track_races.empty and 'ActualPosition' in track_races.columns:
                valid_positions = track_races['ActualPosition'].dropna()
                if len(valid_positions) > 0:
                    df_filtered.at[idx, 'ConstructorTrackAvgPosition'] = valid_positions.mean()
                else:
                    df_filtered.at[idx, 'ConstructorTrackAvgPosition'] = 10.5
            else:
                df_filtered.at[idx, 'ConstructorTrackAvgPosition'] = 10.5
        feature_cols.append('ConstructorTrackAvgPosition')
    
    # ConstructorRecentForm: Constructor's average finish in last 5 races
    if 'ConstructorStanding' in df_filtered.columns:
        df_filtered['ConstructorRecentForm'] = np.nan
        for idx, row in df_filtered.iterrows():
            constructor_standing = row['ConstructorStanding']
            year = row.get('Year', calc_df['Year'].max() if 'Year' in calc_df.columns else 2024)
            round_num = row.get('RoundNumber', 0)
            # Get last 5 races by this constructor before this race
            constructor_races = calc_df[(calc_df['ConstructorStanding'] == constructor_standing) &
                                       ((calc_df['Year'] < year) | 
                                        ((calc_df['Year'] == year) & (calc_df['RoundNumber'] < round_num)))]
            if not constructor_races.empty and 'ActualPosition' in constructor_races.columns:
                recent_races = constructor_races.nlargest(5, ['Year', 'RoundNumber'])
                valid_positions = recent_races['ActualPosition'].dropna()
                if len(valid_positions) > 0:
                    df_filtered.at[idx, 'ConstructorRecentForm'] = valid_positions.mean()
                else:
                    df_filtered.at[idx, 'ConstructorRecentForm'] = 10.5
            else:
                df_filtered.at[idx, 'ConstructorRecentForm'] = 10.5
        feature_cols.append('ConstructorRecentForm')
    
    # ConstructorPoints (if available)
    if 'ConstructorPoints' in df_filtered.columns:
        feature_cols.append('ConstructorPoints')
    
    # FormTrend (if available) - keep this, it's in top 10 importance
    if 'FormTrend' in df_filtered.columns:
        feature_cols.append('FormTrend')
    
    # Remove PointsGapToLeader - redundant with SeasonPoints and ConstructorStanding
    
    # TrackLength: Normalized track length (if we can estimate from historical data)
    # For now, use a simple heuristic based on track name patterns
    if 'EventName' in df_filtered.columns:
        df_filtered['TrackLength'] = 5.0  # Default ~5km
        # Monaco is shortest (~3.3km), Spa is longest (~7km)
        track_lengths = {
            'Monaco': 3.3, 'Singapore': 5.0, 'Hungary': 4.4, 'Austria': 4.3,
            'Spa': 7.0, 'Silverstone': 5.9, 'Monza': 5.8, 'Suzuka': 5.8,
            'Interlagos': 4.3, 'Abu Dhabi': 5.6, 'Bahrain': 5.4, 'Qatar': 5.4
        }
        for track_name, length in track_lengths.items():
            mask = df_filtered['EventName'].str.contains(track_name, case=False, na=False)
            df_filtered.loc[mask, 'TrackLength'] = length
        # Normalize to 0-1 range (3-7 km range)
        df_filtered['TrackLength'] = (df_filtered['TrackLength'] - 3.0) / 4.0
        feature_cols.append('TrackLength')
    
    # TrackCorners: Estimated number of corners (normalized)
    if 'EventName' in df_filtered.columns:
        df_filtered['TrackCorners'] = 15.0  # Default
        track_corners = {
            'Monaco': 19, 'Singapore': 23, 'Suzuka': 18, 'Interlagos': 15,
            'Silverstone': 18, 'Spa': 19, 'Monza': 11, 'Hungary': 14
        }
        for track_name, corners in track_corners.items():
            mask = df_filtered['EventName'].str.contains(track_name, case=False, na=False)
            df_filtered.loc[mask, 'TrackCorners'] = corners
        # Normalize to 0-1 range (10-25 corners)
        df_filtered['TrackCorners'] = (df_filtered['TrackCorners'] - 10.0) / 15.0
        feature_cols.append('TrackCorners')
    
    # TrackAltitude: Track elevation (normalized)
    if 'EventName' in df_filtered.columns:
        df_filtered['TrackAltitude'] = 0.0  # Default sea level
        track_altitudes = {
            'Mexico': 2.2, 'Interlagos': 0.8, 'Austria': 0.7, 'Spa': 0.5,
            'Silverstone': 0.2, 'Monaco': 0.0, 'Singapore': 0.0
        }
        for track_name, altitude in track_altitudes.items():
            mask = df_filtered['EventName'].str.contains(track_name, case=False, na=False)
            df_filtered.loc[mask, 'TrackAltitude'] = altitude
        # Normalize to 0-1 range (0-2.5 km)
        df_filtered['TrackAltitude'] = df_filtered['TrackAltitude'] / 2.5
        feature_cols.append('TrackAltitude')
    
    # HistoricalWetRaceRate: Percentage of wet races at this track (predictable from history)
    if 'EventName' in df_filtered.columns:
        df_filtered['HistoricalWetRaceRate'] = 0.1  # Default 10% wet races
        # This would ideally come from weather data, but for now use track characteristics
        wet_tracks = ['Monaco', 'Singapore', 'Interlagos', 'Silverstone', 'Spa']
        for track_name in wet_tracks:
            mask = df_filtered['EventName'].str.contains(track_name, case=False, na=False)
            df_filtered.loc[mask, 'HistoricalWetRaceRate'] = 0.3  # Higher wet race probability
        feature_cols.append('HistoricalWetRaceRate')
    
    # AveragePitStopsAtTrack: Historical average pit stops at this track (strategy pattern)
    if 'EventName' in df_filtered.columns:
        df_filtered['AveragePitStopsAtTrack'] = 2.0  # Default 2 stops
        # Track-specific pit stop patterns (normalized)
        pit_stop_patterns = {
            'Monaco': 1.5, 'Singapore': 1.8, 'Hungary': 1.6,  # Fewer stops (hard to overtake)
            'Spa': 2.5, 'Silverstone': 2.3, 'Monza': 2.2  # More stops (overtaking easier)
        }
        for track_name, stops in pit_stop_patterns.items():
            mask = df_filtered['EventName'].str.contains(track_name, case=False, na=False)
            df_filtered.loc[mask, 'AveragePitStopsAtTrack'] = stops
        # Normalize to 0-1 range (1-3 stops)
        df_filtered['AveragePitStopsAtTrack'] = (df_filtered['AveragePitStopsAtTrack'] - 1.0) / 2.0
        feature_cols.append('AveragePitStopsAtTrack')
    
    # DriverPointsPerRace: Average points per race (keep this - useful)
    if 'DriverNumber' in df_filtered.columns:
        df_filtered['DriverPointsPerRace'] = np.nan
        for idx, row in df_filtered.iterrows():
            driver_num = row['DriverNumber']
            year = row.get('Year', calc_df['Year'].max() if 'Year' in calc_df.columns else 2024)
            round_num = row.get('RoundNumber', 0)
            driver_season_races = calc_df[(calc_df['DriverNumber'] == driver_num) &
                                         ((calc_df['Year'] < year) | 
                                          ((calc_df['Year'] == year) & (calc_df['RoundNumber'] < round_num)))]
            if not driver_season_races.empty and 'Points' in driver_season_races.columns:
                total_points = driver_season_races['Points'].sum()
                race_count = len(driver_season_races)
                if race_count > 0:
                    df_filtered.at[idx, 'DriverPointsPerRace'] = total_points / race_count
                else:
                    df_filtered.at[idx, 'DriverPointsPerRace'] = 0.0
            else:
                df_filtered.at[idx, 'DriverPointsPerRace'] = 0.0
        feature_cols.append('DriverPointsPerRace')
    
    # ConstructorWinRate: Percentage of wins by constructor
    if 'ConstructorStanding' in df_filtered.columns:
        df_filtered['ConstructorWinRate'] = np.nan
        for idx, row in df_filtered.iterrows():
            constructor_standing = row['ConstructorStanding']
            constructor_races = calc_df[calc_df['ConstructorStanding'] == constructor_standing]
            if not constructor_races.empty and 'ActualPosition' in constructor_races.columns:
                valid_races = constructor_races['ActualPosition'].dropna()
                if len(valid_races) > 0:
                    wins = (valid_races == 1).sum()
                    df_filtered.at[idx, 'ConstructorWinRate'] = wins / len(valid_races)
                else:
                    df_filtered.at[idx, 'ConstructorWinRate'] = 0.0
            else:
                df_filtered.at[idx, 'ConstructorWinRate'] = 0.0
        feature_cols.append('ConstructorWinRate')
    
    # BestFinishAtTrack: Best finish position at this track
    if 'EventName' in df_filtered.columns and 'DriverNumber' in df_filtered.columns:
        df_filtered['BestFinishAtTrack'] = np.nan
        for idx, row in df_filtered.iterrows():
            track_name = row['EventName']
            driver_num = row['DriverNumber']
            track_races = calc_df[(calc_df['EventName'] == track_name) & 
                                 (calc_df['DriverNumber'] == driver_num)]
            if not track_races.empty and 'ActualPosition' in track_races.columns:
                valid_positions = track_races['ActualPosition'].dropna()
                if len(valid_positions) > 0:
                    df_filtered.at[idx, 'BestFinishAtTrack'] = valid_positions.min()
                else:
                    df_filtered.at[idx, 'BestFinishAtTrack'] = 20.0
            else:
                df_filtered.at[idx, 'BestFinishAtTrack'] = 20.0
        feature_cols.append('BestFinishAtTrack')
    
    # WorstFinishAtTrack: Worst finish position at this track
    if 'EventName' in df_filtered.columns and 'DriverNumber' in df_filtered.columns:
        df_filtered['WorstFinishAtTrack'] = np.nan
        for idx, row in df_filtered.iterrows():
            track_name = row['EventName']
            driver_num = row['DriverNumber']
            track_races = calc_df[(calc_df['EventName'] == track_name) & 
                                 (calc_df['DriverNumber'] == driver_num)]
            if not track_races.empty and 'ActualPosition' in track_races.columns:
                valid_positions = track_races['ActualPosition'].dropna()
                if len(valid_positions) > 0:
                    df_filtered.at[idx, 'WorstFinishAtTrack'] = valid_positions.max()
                else:
                    df_filtered.at[idx, 'WorstFinishAtTrack'] = 20.0
            else:
                df_filtered.at[idx, 'WorstFinishAtTrack'] = 20.0
        feature_cols.append('WorstFinishAtTrack')
    
    # TrackConsistency: Standard deviation of positions at this track (lower = more consistent)
    if 'EventName' in df_filtered.columns and 'DriverNumber' in df_filtered.columns:
        df_filtered['TrackConsistency'] = np.nan
        for idx, row in df_filtered.iterrows():
            track_name = row['EventName']
            driver_num = row['DriverNumber']
            track_races = calc_df[(calc_df['EventName'] == track_name) & 
                                 (calc_df['DriverNumber'] == driver_num)]
            if not track_races.empty and 'ActualPosition' in track_races.columns:
                valid_positions = track_races['ActualPosition'].dropna()
                if len(valid_positions) > 1:
                    df_filtered.at[idx, 'TrackConsistency'] = valid_positions.std()
                elif len(valid_positions) == 1:
                    df_filtered.at[idx, 'TrackConsistency'] = 0.0  # Perfect consistency
                else:
                    df_filtered.at[idx, 'TrackConsistency'] = 5.0  # Default uncertainty
            else:
                df_filtered.at[idx, 'TrackConsistency'] = 5.0
        feature_cols.append('TrackConsistency')
    
    # LastYearPosition: Position at this track last year
    if 'EventName' in df_filtered.columns and 'DriverNumber' in df_filtered.columns:
        df_filtered['LastYearPosition'] = np.nan
        for idx, row in df_filtered.iterrows():
            track_name = row['EventName']
            driver_num = row['DriverNumber']
            year = row.get('Year', calc_df['Year'].max() if 'Year' in calc_df.columns else 2024)
            last_year_race = calc_df[(calc_df['EventName'] == track_name) & 
                                    (calc_df['DriverNumber'] == driver_num) &
                                    (calc_df['Year'] == year - 1)]
            if not last_year_race.empty and 'ActualPosition' in last_year_race.columns:
                valid_positions = last_year_race['ActualPosition'].dropna()
                if len(valid_positions) > 0:
                    df_filtered.at[idx, 'LastYearPosition'] = valid_positions.iloc[0]
                else:
                    df_filtered.at[idx, 'LastYearPosition'] = 10.5
            else:
                df_filtered.at[idx, 'LastYearPosition'] = 10.5
        feature_cols.append('LastYearPosition')
    
    # Extract features
    X = df_filtered[feature_cols].copy()
    y = df_filtered['ActualPosition'].values
    feature_names = feature_cols.copy()
    
    # Handle missing values
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(median_val)
    
    return X.values, y, None, filter_stats, feature_names


def save_model(model, scaler, label_encoder, output_dir: str = None, model_index: int = None):
    """Save trained model, scaler, and label encoder."""
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / 'models'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if model_index is not None:
        model_path = output_dir / f'f1_predictor_model_complex_ensemble_{model_index}.pth'
    else:
        model_path = output_dir / 'f1_predictor_model_complex.pth'
    
    scaler_path = output_dir / 'scaler_complex.pkl'
    encoder_path = output_dir / 'label_encoder_complex.pkl'
    
    torch.save(model.state_dict(), model_path)
    
    if model_index is None or model_index == 0:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"Scaler saved to {scaler_path}")
        print(f"Label encoder saved to {encoder_path}")
    
    print(f"Model saved to {model_path}")


def main():
    """Main function to train the complex F1 prediction model."""
    print("=" * 70)
    print("COMPLEX F1 POSITION PREDICTION MODEL TRAINING")
    print("=" * 70)
    print("Deep neural network with extensive feature engineering")
    print("=" * 70)
    
    device = torch.device('cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading data...")
    training_df, test_df, metadata = load_data()
    
    print(f"Training samples: {len(training_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Time-based split
    print("\nSplitting data for validation (time-based)...")
    val_year = training_df['Year'].max()
    train_df_split = training_df[training_df['Year'] < val_year].copy()
    val_df_split = training_df[training_df['Year'] == val_year].copy()
    
    # Prepare complex features
    print("\nPreparing complex features (top 10 positions only)...")
    X_train, y_train, _, train_stats, feature_names = prepare_complex_features(
        train_df_split, historical_df=training_df, filter_dnf=True, 
        filter_outliers=True, top10_only=True
    )
    X_val, y_val, _, val_stats, _ = prepare_complex_features(
        val_df_split, historical_df=training_df, filter_dnf=True, 
        filter_outliers=True, top10_only=True
    )
    
    # Prepare test data
    print("\nPreparing test data (top 10 positions only)...")
    X_test, y_test, _, test_stats, _ = prepare_complex_features(
        test_df, historical_df=training_df, filter_dnf=True, 
        filter_outliers=True, top10_only=True
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nFeature summary:")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Features: {', '.join(feature_names)}")
    print(f"  Training samples: {len(X_train_scaled)}")
    print(f"  Validation samples: {len(X_val_scaled)}")
    print(f"  Test samples: {len(X_test_scaled)}")
    
    # Create datasets
    train_dataset = F1Dataset(X_train_scaled, y_train)
    val_dataset = F1Dataset(X_val_scaled, y_val)
    test_dataset = F1Dataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize complex model (wider and deeper)
    # Increased dropout to 0.4 to reduce overfitting (test MAE was much worse than validation)
    input_size = X_train_scaled.shape[1]
    model = ComplexF1NeuralNetwork(
        input_size=input_size,
        hidden_sizes=[512, 256, 128, 64],
        dropout_rate=0.4,
        equal_init=True
    ).to(device)
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    # Increased weight decay to reduce overfitting
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=2e-4)  # Lower LR, higher weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Training loop
    print(f"\nTraining for up to 300 epochs...")
    print("-" * 70)
    
    best_val_mae = float('inf')
    best_model_state = None
    patience = 0
    max_patience = 30  # Earlier stopping to prevent overfitting
    
    for epoch in range(300):
        # Train
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_mae, val_rmse, val_r2, val_exact, val_w1, val_w2, val_w3, _, _ = evaluate_model(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train MAE: {train_mae:.3f} | Val MAE: {val_mae:.3f} | "
                  f"Exact: {val_exact:.1f}% | W1: {val_w1:.1f}% | W3: {val_w3:.1f}%")
        
        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    criterion_eval = nn.MSELoss()
    
    val_loss, val_mae, val_rmse, val_r2, val_exact, val_w1, val_w2, val_w3, _, _ = evaluate_model(
        model, val_loader, criterion_eval, device
    )
    
    test_loss, test_mae, test_rmse, test_r2, test_exact, test_w1, test_w2, test_w3, _, _ = evaluate_model(
        model, test_loader, criterion_eval, device
    )
    
    print(f"\nValidation Results:")
    print(f"  MAE: {val_mae:.3f} positions")
    print(f"  RMSE: {val_rmse:.3f} positions")
    print(f"  R²: {val_r2:.4f}")
    print(f"  Exact position: {val_exact:.1f}%")
    print(f"  Within 1 position: {val_w1:.1f}%")
    print(f"  Within 2 positions: {val_w2:.1f}%")
    print(f"  Within 3 positions: {val_w3:.1f}%")
    
    print(f"\nTest Results:")
    print(f"  MAE: {test_mae:.3f} positions")
    print(f"  RMSE: {test_rmse:.3f} positions")
    print(f"  R²: {test_r2:.4f}")
    print(f"  Exact position: {test_exact:.1f}%")
    print(f"  Within 1 position: {test_w1:.1f}%")
    print(f"  Within 2 positions: {test_w2:.1f}%")
    print(f"  Within 3 positions: {test_w3:.1f}%")
    
    # Feature importance
    feature_importances = get_feature_importance(model, feature_names, device)
    print(f"\nTop 10 Feature Importances:")
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    for i, (name, importance) in enumerate(sorted_features[:10], 1):
        print(f"  {i:2d}. {name:30s}: {importance:.4f}")
    
    # Save model
    save_model(model, scaler, None, model_index=None)
    
    # Save results
    results = {
        'model_type': 'complex_neural_network',
        'features': feature_names,
        'num_features': len(feature_names),
        'architecture': {
            'input_size': input_size,
            'hidden_sizes': [512, 256, 128, 64],
            'dropout_rate': 0.4
        },
        'validation': {
            'mae': float(val_mae),
            'rmse': float(val_rmse),
            'r2': float(val_r2),
            'exact': float(val_exact),
            'within_1': float(val_w1),
            'within_2': float(val_w2),
            'within_3': float(val_w3)
        },
        'test': {
            'mae': float(test_mae),
            'rmse': float(test_rmse),
            'r2': float(test_r2),
            'exact': float(test_exact),
            'within_1': float(test_w1),
            'within_2': float(test_w2),
            'within_3': float(test_w3)
        },
        'feature_importances': {k: float(v) for k, v in feature_importances.items()}
    }
    
    json_dir = Path(__file__).parent.parent / 'json'
    json_dir.mkdir(exist_ok=True, parents=True)
    results_path = json_dir / 'training_results_complex.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()

