"""
Inference script for F1 Predictions (Top 10 Model).
Loads trained top 10 model (trained on positions 1-10 only) and makes predictions for race positions.
Ranks drivers and displays predicted top 10.
Predictions are clipped to range 1-10 to match training data.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import torch
import torch.nn as nn
# Lazy import fastf1 - only import when needed for future race selection

# Feature columns (7 base features)
FEATURE_COLS = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                'ConstructorStanding', 'GridPosition', 'RecentForm', 'TrackType']


def get_status(error: int) -> str:
    """Get status string based on error."""
    if error == 0:
        return "Exact"
    elif error == 1:
        return "Close"
    elif error == 2:
        return "Good"
    elif error <= 3:
        return "Fair"
    else:
        return "Poor"


def handle_nan_values(X: np.ndarray) -> np.ndarray:
    """
    Handle NaN values in feature matrix.
    For GridPosition specifically, we use per-driver historical averages,
    so if all values are NaN, we use a reasonable default (10.5 = mid-field)
    instead of filling with the same mean for all drivers.
    """
    with np.errstate(all='ignore'):
        col_means = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            if np.isnan(col_means[i]) or np.isinf(col_means[i]):
                median_val = np.nanmedian(X[:, i])
                fill_val = 0 if np.isnan(median_val) else median_val
                X[:, i] = np.nan_to_num(X[:, i], nan=fill_val)
            else:
                # Check if all values in this column are NaN
                nan_mask = np.isnan(X[:, i])
                if nan_mask.all():
                    # All NaN - use a reasonable default (10.5 for grid position, mid-field)
                    # This should rarely happen if calculate_future_race_features works correctly
                    X[:, i] = 10.5
                else:
                    # Some NaN - fill with mean of non-NaN values
                    X[:, i] = np.nan_to_num(X[:, i], nan=col_means[i])
    return X


def make_predictions(X_scaled: np.ndarray, model, model_type: str, device=None) -> np.ndarray:
    """Make predictions using model."""
    if model_type == 'neural_network':
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            predictions = model(X_tensor).cpu().numpy()
            # Ensure predictions are 1D array (flatten if needed)
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            return predictions
    else:
        return model.predict(X_scaled)


class F1NeuralNetwork(nn.Module):
    """Neural Network model definition (must match training - regression)."""
    
    def __init__(self, input_size=7, hidden_sizes=[256, 128, 64], dropout_rate=0.4):
        super(F1NeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.network(x)
        # Squeeze only the last dimension (output dimension), not batch dimension
        if output.dim() > 1:
            return output.squeeze(-1)  # Squeeze last dimension only
        return output.squeeze()


def load_model(model_dir: str = None, model_type: str = 'neural_network', auto_fallback: bool = True):
    """
    Load trained model and scaler.
    
    Args:
        model_dir: Directory containing model files (default: ../models relative to script)
        model_type: 'neural_network' or 'random_forest'
        auto_fallback: If True, try the other model type if requested one is not found
        
    Returns:
        Tuple of (model, scaler, model_type, device)
    """
    if model_dir is None:
        # Resolve path relative to this script's location
        script_dir = Path(__file__).parent
        model_dir = script_dir.parent / 'models'
    else:
        model_dir = Path(model_dir)
    
    if model_type == 'neural_network':
        nn_model_path = model_dir / 'f1_predictor_model_top10.pth'
        nn_scaler_path = model_dir / 'scaler_top10.pkl'
        
        if nn_model_path.exists() and nn_scaler_path.exists():
            # Load scaler first to determine input size
            with open(nn_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Get input size from scaler (should be 7 base features)
            input_size = getattr(scaler, 'n_features_in_', 7)
            
            # Initialize and load model with correct input size
            device = torch.device('cpu')
            model = F1NeuralNetwork(
                input_size=input_size,
                hidden_sizes=[256, 128, 64],  # Architecture optimized for winner prediction
                dropout_rate=0.4
            ).to(device)
            model.load_state_dict(torch.load(nn_model_path, map_location=device))
            model.eval()
            
            return model, scaler, 'neural_network', device
        elif auto_fallback:
            # Try Random Forest as fallback
            print(f"Warning: Neural network model not found at {nn_model_path}")
            print("Attempting to load Random Forest model instead...")
            model_type = 'random_forest'  # Switch to try RF
        else:
            raise FileNotFoundError(f"Model not found at {nn_model_path}. Run top10/train.py first.")
    
    # Try Random Forest (either requested or as fallback)
    if model_type == 'random_forest':
        rf_model_path = model_dir / 'f1_predictor_model_rf.pkl'
        rf_scaler_path = model_dir / 'scaler_rf.pkl'
        
        if rf_model_path.exists() and rf_scaler_path.exists():
            with open(rf_model_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(rf_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            return model, scaler, 'random_forest', None
        elif auto_fallback:
            raise FileNotFoundError(
                f"Neither model found!\n"
                f"  Neural Network: {Path(model_dir) / 'f1_predictor_model.pth'} (not found)\n"
                f"  Random Forest: {rf_model_path} (not found)\n"
                f"Please train at least one model:\n"
                f"  python train.py (for neural network)\n"
                f"  python train_rf.py (for random forest)"
            )
        else:
            raise FileNotFoundError(f"Model not found at {rf_model_path}. Run train_rf.py first.")
    
    raise ValueError(f"Unknown model type: {model_type}")


def predict_position(season_points: float, season_avg_finish: float, 
                    historical_track_avg: float, constructor_points: float,
                    constructor_standing: float, grid_position: float,
                    model, scaler, model_type='neural_network', device=None,
                    recent_form: float = None):
    """
    Predict finishing position for a driver given input features.
    
    Args:
        season_points: Season points accumulated so far
        season_avg_finish: Average finish position this season
        historical_track_avg: Historical average position at this track
        constructor_points: Constructor's total points
        constructor_standing: Constructor's championship standing
        grid_position: Average grid position (historical average, matches training data)
        recent_form: Average finish position in last 5 races (optional)
        model: Trained model
        scaler: Fitted scaler
        model_type: 'neural_network' or 'random_forest'
        device: Device for neural network inference
        
    Returns:
        Predicted finishing position (1-20)
    """
    # Use 7 features (ConstructorPoints removed for better accuracy)
    # Note: This function is deprecated - use predict_race_top10 instead which handles TrackType
    # TrackType defaults to 0 (permanent circuit) if not provided
    if recent_form is None:
        recent_form = np.nan  # Use NaN if not provided
    track_type = 0  # Default to permanent circuit if not provided
    features = np.array([[season_points, season_avg_finish, historical_track_avg,
                          constructor_standing, grid_position, recent_form, track_type]])
    
    # Handle NaN values
    features = np.nan_to_num(features, nan=np.nanmedian(features, axis=0))
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    if model_type == 'neural_network':
        # Neural network prediction
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(device)
            predicted_position = model(features_tensor).cpu().item()
    else:
        # Random Forest prediction
        predicted_position = model.predict(features_scaled)[0]
    
    # Ensure position is within valid range (1-10, matching top 10 training data)
    predicted_position = max(1, min(10, predicted_position))
    
    return predicted_position


def predict_race_top10(drivers_df: pd.DataFrame, model, scaler, 
                       model_type='neural_network', device=None):
    """
    Predict positions for all drivers in a race and return top 10.
    
    Args:
        drivers_df: DataFrame with one row per driver, containing:
                   ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                    'ConstructorStanding', 'GridPosition', 'RecentForm', 'TrackType', 'DriverNumber', 'DriverName']
        model: Trained model
        scaler: Fitted scaler
        model_type: 'neural_network' or 'random_forest'
        device: Device for neural network inference
        
    Returns:
        DataFrame with predicted positions, sorted to show top 10
    """
    # Check if required columns exist
    missing_cols = [col for col in FEATURE_COLS if col not in drivers_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Prepare features
    X = drivers_df[FEATURE_COLS].values
    X = handle_nan_values(X)
    X_scaled = scaler.transform(X)
    predicted_positions = make_predictions(X_scaled, model, model_type, device)
    
    # Ensure positions are in valid range (1-10, matching top 10 training data)
    predicted_positions = np.clip(predicted_positions, 1, 10)
    
    # Add predictions to DataFrame
    result_df = drivers_df.copy()
    result_df['PredictedPosition'] = predicted_positions
    
    # Sort by predicted position (best first)
    result_df = result_df.sort_values('PredictedPosition')
    
    # Add rank
    result_df['Rank'] = range(1, len(result_df) + 1)
    
    # Return top 10
    top10 = result_df.head(10).copy()
    
    return top10, result_df


def predict_from_dataframe(df: pd.DataFrame, model, scaler,
                          model_type='neural_network', device=None):
    """
    Predict positions for multiple samples from a DataFrame.
    
    Args:
        df: DataFrame with feature columns
        model: Trained model
        scaler: Fitted scaler
        model_type: 'neural_network' or 'random_forest'
        device: Device for neural network inference
        
    Returns:
        DataFrame with predictions added
    """
    # Check if required columns exist
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Prepare features
    X = df[FEATURE_COLS].values
    X = handle_nan_values(X)
    X_scaled = scaler.transform(X)
    predicted_positions = make_predictions(X_scaled, model, model_type, device)
    
    # Ensure positions are in valid range (1-10, matching top 10 training data)
    predicted_positions = np.clip(predicted_positions, 1, 10)
    
    # Add predictions to DataFrame
    df['PredictedPosition'] = predicted_positions
    
    return df


def calculate_future_race_features(test_df: pd.DataFrame, selected_year: int, selected_round: int, 
                                    track_name: str, training_df: pd.DataFrame = None):
    """
    Calculate features for a future race using data from all completed races up to this point.
    
    Args:
        test_df: DataFrame with test data (completed races)
        selected_year: Year of the future race
        selected_round: Round number of the future race
        track_name: Name of the track for the future race
        
    Returns:
        DataFrame with driver features for the future race
    """
    # Get all completed races up to (but not including) the future race
    completed_races = test_df[
        (test_df['Year'] == selected_year) & 
        (test_df['RoundNumber'] < selected_round)
    ].copy()
    
    if completed_races.empty:
        # If no races in this year yet, use last year's final data
        last_year_races = test_df[test_df['Year'] < selected_year]
        if not last_year_races.empty:
            last_year = last_year_races['Year'].max()
            completed_races = test_df[test_df['Year'] == last_year].copy()
    
    if completed_races.empty:
        raise ValueError("No historical data available to calculate features for future race")
    
    # Get the most recent completed race for driver list and initial state
    most_recent_round = completed_races['RoundNumber'].max()
    most_recent_race = completed_races[completed_races['RoundNumber'] == most_recent_round].copy()
    
    print(f"  Calculating features from {len(completed_races)} completed races (most recent: Round {most_recent_round})")
    
    # For each driver in the most recent race, use their current features
    # These features represent their state up to the most recent race
    future_race_features = []
    
    # Calculate track-specific historical averages for this specific track
    # Use training data if available (has more historical data), otherwise use test data
    historical_df = training_df if training_df is not None and not training_df.empty else test_df

    track_historical_data = historical_df[historical_df['EventName'] == track_name]
    track_avg_by_driver = {}
    grid_avg_by_driver = {}  # Average grid position per driver
    
    for driver_num in track_historical_data['DriverNumber'].unique():
        driver_track_races = track_historical_data[track_historical_data['DriverNumber'] == driver_num]
        # Try to get ActualPosition or Position column
        if 'ActualPosition' in driver_track_races.columns:
            valid_positions = driver_track_races['ActualPosition'].dropna()
        elif 'Position' in driver_track_races.columns:
            valid_positions = driver_track_races['Position'].dropna()
        else:
            valid_positions = pd.Series()
        
        if len(valid_positions) > 0:
            track_avg_by_driver[str(driver_num)] = valid_positions.mean()
        else:
            # Fallback: use driver's overall average from historical data
            driver_all_races = historical_df[historical_df['DriverNumber'] == driver_num]
            if not driver_all_races.empty:
                if 'ActualPosition' in driver_all_races.columns:
                    valid_positions = driver_all_races['ActualPosition'].dropna()
                elif 'Position' in driver_all_races.columns:
                    valid_positions = driver_all_races['Position'].dropna()
                else:
                    valid_positions = pd.Series()
                
                if len(valid_positions) > 0:
                    track_avg_by_driver[str(driver_num)] = valid_positions.mean()
                else:
                    track_avg_by_driver[str(driver_num)] = 10.0  # Default for rookies
            else:
                track_avg_by_driver[str(driver_num)] = 10.0  # Default for rookies
        
        # Calculate average grid position for this driver (from all historical races)
        driver_all_races = historical_df[historical_df['DriverNumber'] == driver_num]
        if not driver_all_races.empty and 'GridPosition' in driver_all_races.columns:
            valid_grid_positions = driver_all_races['GridPosition'].dropna()
            if len(valid_grid_positions) > 0:
                grid_avg_by_driver[str(driver_num)] = valid_grid_positions.mean()
            else:
                grid_avg_by_driver[str(driver_num)] = np.nan
        else:
            grid_avg_by_driver[str(driver_num)] = np.nan
    
    # F1 points system for calculating cumulative points
    points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    
    for _, driver_row in most_recent_race.iterrows():
        driver_num = driver_row['DriverNumber']
        driver_name = driver_row.get('DriverName', f"Driver {driver_num}")
        
        # Calculate cumulative features from ALL completed races (not just most recent)
        driver_completed_races = completed_races[completed_races['DriverNumber'] == driver_num].copy()
        
        # Calculate SeasonPoints: sum of all points from completed races
        if not driver_completed_races.empty:
            # Prefer Points column if it exists (more accurate)
            if 'Points' in driver_completed_races.columns:
                season_points = driver_completed_races['Points'].sum()
            elif 'ActualPosition' in driver_completed_races.columns:
                # Calculate points from actual positions
                total_points = 0
                for _, race in driver_completed_races.iterrows():
                    pos = race.get('ActualPosition', np.nan)
                    if not pd.isna(pos) and 1 <= pos <= 10:
                        total_points += points_system.get(int(pos), 0)
                season_points = total_points
            else:
                # Fallback: use most recent race's cumulative points
                season_points = driver_row.get('SeasonPoints', 0)
        else:
            # Fallback: use most recent race's cumulative points
            season_points = driver_row.get('SeasonPoints', 0)
        
        # Calculate SeasonAvgFinish: average of all completed race positions
        if not driver_completed_races.empty and 'ActualPosition' in driver_completed_races.columns:
            valid_positions = driver_completed_races['ActualPosition'].dropna()
            if len(valid_positions) > 0:
                season_avg_finish = valid_positions.mean()
            else:
                season_avg_finish = driver_row.get('SeasonAvgFinish', np.nan)
        else:
            season_avg_finish = driver_row.get('SeasonAvgFinish', np.nan)
        
        # Calculate RecentForm: average of last 5 completed races
        if not driver_completed_races.empty and 'ActualPosition' in driver_completed_races.columns:
            # Sort by round number descending to get most recent
            driver_races_sorted = driver_completed_races.sort_values('RoundNumber', ascending=False)
            last_5_races = driver_races_sorted.head(5)
            valid_positions = last_5_races['ActualPosition'].dropna()
            if len(valid_positions) > 0:
                recent_form = valid_positions.mean()
            else:
                recent_form = driver_row.get('RecentForm', np.nan)
        else:
            recent_form = driver_row.get('RecentForm', np.nan)
        
        # Calculate ConstructorPoints and ConstructorStanding from completed races
        # Group by constructor (simplified - use most recent race's values)
        constructor_points = driver_row.get('ConstructorPoints', 0)
        constructor_standing = driver_row.get('ConstructorStanding', 10)
        
        # Get track-specific historical average for this driver at this track
        # Default to 10.0 for rookies/drivers with no historical data
        hist_track_avg = track_avg_by_driver.get(str(driver_num), 
                                                  driver_row.get('HistoricalTrackAvgPosition', 10.0))
        if pd.isna(hist_track_avg):
            hist_track_avg = 10.0
        
        # Get driver's historical average grid position (or use most recent if no history)
        driver_grid_avg = grid_avg_by_driver.get(str(driver_num), np.nan)
        if pd.isna(driver_grid_avg):
            # Fallback: use most recent grid position, or historical average from driver_row
            driver_grid_avg = driver_row.get('GridPosition', np.nan)
            if pd.isna(driver_grid_avg):
                # Last resort: use driver's overall average from historical data
                driver_all_races = historical_df[historical_df['DriverNumber'] == driver_num]
                if not driver_all_races.empty and 'GridPosition' in driver_all_races.columns:
                    valid_grid = driver_all_races['GridPosition'].dropna()
                    if len(valid_grid) > 0:
                        driver_grid_avg = valid_grid.mean()
        
        # Ensure we always have a valid GridPosition (default to 10.5 = mid-field if no history)
        if pd.isna(driver_grid_avg):
            driver_grid_avg = 10.5  # Mid-field default for drivers with no grid history
        
        # Calculate TrackType (street circuit = 1, permanent = 0)
        street_circuits = [
            'Monaco', 'Singapore', 'Azerbaijan', 'Miami', 'Las Vegas',
            'Saudi Arabian'
        ]
        track_type = 1 if any(street in track_name for street in street_circuits) else 0
        
        # Use calculated cumulative features
        features = {
            'Year': selected_year,
            'EventName': track_name,
            'RoundNumber': selected_round,
            'SeasonPoints': season_points,  # Calculated from all completed races
            'SeasonAvgFinish': season_avg_finish,  # Calculated from all completed races
            'HistoricalTrackAvgPosition': hist_track_avg if not pd.isna(hist_track_avg) else 10.0,  # Track-specific average (default 10.0 for rookies)
            'ConstructorPoints': constructor_points,
            'ConstructorStanding': constructor_standing,
            'GridPosition': driver_grid_avg if not pd.isna(driver_grid_avg) else 10.5,  # Use driver's historical average grid position (default 10.5 if no history)
            'RecentForm': recent_form,  # Calculated from last 5 completed races
            'TrackType': track_type,  # Street circuit (1) or permanent (0)
            'DriverNumber': driver_num,
            'DriverName': driver_name,
            'ActualPosition': np.nan  # Future race, no actual position
        }
        future_race_features.append(features)
    
    return pd.DataFrame(future_race_features)


def update_future_race_features_progressive(race_df: pd.DataFrame, previous_state_df: pd.DataFrame, 
                                           _race_name: str) -> pd.DataFrame:
    """
    Update features for a future race based on previous race's state.
    This allows progressive feature updates across multiple future races.
    """
    updated_df = race_df.copy()
    
    # Update features from previous state
    for idx, row in updated_df.iterrows():
        driver_num = row['DriverNumber']
        prev_driver_data = previous_state_df[previous_state_df['DriverNumber'] == driver_num]
        
        if not prev_driver_data.empty:
            prev_row = prev_driver_data.iloc[0]
            # Update features that would change between races
            updated_df.at[idx, 'SeasonPoints'] = prev_row.get('SeasonPoints', row.get('SeasonPoints', 0))
            updated_df.at[idx, 'SeasonAvgFinish'] = prev_row.get('SeasonAvgFinish', row.get('SeasonAvgFinish', np.nan))
            updated_df.at[idx, 'ConstructorPoints'] = prev_row.get('ConstructorPoints', row.get('ConstructorPoints', 0))
            updated_df.at[idx, 'ConstructorStanding'] = prev_row.get('ConstructorStanding', row.get('ConstructorStanding', 10))
            updated_df.at[idx, 'RecentForm'] = prev_row.get('RecentForm', row.get('RecentForm', np.nan))
            # Preserve GridPosition from current row (it's driver-specific historical average, shouldn't change)
            # If it's NaN in current row but exists in previous, use previous as fallback
            if pd.isna(row.get('GridPosition', np.nan)) and not pd.isna(prev_row.get('GridPosition', np.nan)):
                updated_df.at[idx, 'GridPosition'] = prev_row.get('GridPosition')
            # TrackType doesn't change (it's track-specific), so keep it from current row
    
    return updated_df


def recalculate_features_from_state(race_df: pd.DataFrame, previous_state_df: pd.DataFrame, 
                                    track_name: str, training_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Recalculate ALL features for a future race based on the updated state from previous future race.
    This ensures features properly reflect all previous races including simulated future ones.
    """
    updated_df = race_df.copy()
    
    # Get track-specific historical averages
    historical_df = training_df if training_df is not None and not training_df.empty else None
    track_avg_by_driver = {}
    if historical_df is not None:
        track_historical_data = historical_df[historical_df['EventName'] == track_name]
        for driver_num in track_historical_data['DriverNumber'].unique():
            driver_track_races = track_historical_data[track_historical_data['DriverNumber'] == driver_num]
            if 'ActualPosition' in driver_track_races.columns:
                valid_positions = driver_track_races['ActualPosition'].dropna()
                if len(valid_positions) > 0:
                    track_avg_by_driver[str(driver_num)] = valid_positions.mean()
    
    # Calculate TrackType
    street_circuits = [
        'Monaco', 'Singapore', 'Azerbaijan', 'Miami', 'Las Vegas',
        'Saudi Arabian'
    ]
    track_type = 1 if any(street in track_name for street in street_circuits) else 0
    
    # Update features from previous state for each driver
    for idx, row in updated_df.iterrows():
        driver_num = row['DriverNumber']
        prev_driver_data = previous_state_df[previous_state_df['DriverNumber'] == driver_num]
        
        if not prev_driver_data.empty:
            prev_row = prev_driver_data.iloc[0]
            # Update all features from previous state
            updated_df.at[idx, 'SeasonPoints'] = prev_row.get('SeasonPoints', row.get('SeasonPoints', 0))
            updated_df.at[idx, 'SeasonAvgFinish'] = prev_row.get('SeasonAvgFinish', row.get('SeasonAvgFinish', np.nan))
            updated_df.at[idx, 'ConstructorPoints'] = prev_row.get('ConstructorPoints', row.get('ConstructorPoints', 0))
            updated_df.at[idx, 'ConstructorStanding'] = prev_row.get('ConstructorStanding', row.get('ConstructorStanding', 10))
            updated_df.at[idx, 'RecentForm'] = prev_row.get('RecentForm', row.get('RecentForm', np.nan))
            
            # GridPosition should be preserved (driver-specific historical average)
            if pd.isna(row.get('GridPosition', np.nan)) and not pd.isna(prev_row.get('GridPosition', np.nan)):
                updated_df.at[idx, 'GridPosition'] = prev_row.get('GridPosition')
            elif pd.isna(row.get('GridPosition', np.nan)):
                updated_df.at[idx, 'GridPosition'] = 10.5  # Default if still missing
            
            # HistoricalTrackAvgPosition - use track-specific if available
            hist_track_avg = track_avg_by_driver.get(str(driver_num), prev_row.get('HistoricalTrackAvgPosition', 10.0))
            if pd.isna(hist_track_avg):
                hist_track_avg = 10.0
            updated_df.at[idx, 'HistoricalTrackAvgPosition'] = hist_track_avg
            
            # TrackType is track-specific, set it
            updated_df.at[idx, 'TrackType'] = track_type
    
    return updated_df


def update_state_with_predictions(state_df: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Update state DataFrame with predictions as simulated race results.
    This allows progressive feature calculation across future races.
    Simulates what features would be after this race based on predicted positions.
    """
    updated_state = state_df.copy()
    
    # F1 points system (position -> points)
    points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    
    # Sort predictions by predicted position to get actual race order
    sorted_predictions = predictions_df.sort_values('PredictedPosition').copy()
    
    # Update each driver's features based on their predicted position
    for rank, (_, pred_row) in enumerate(sorted_predictions.iterrows(), 1):
        driver_num = pred_row.get('DriverNumber')
        if driver_num is not None:
            state_idx = updated_state[updated_state['DriverNumber'] == driver_num].index
            if len(state_idx) > 0:
                idx = state_idx[0]
                pred_pos = pred_row.get('PredictedPosition', rank)
                actual_pos = int(round(pred_pos))  # Round to nearest integer for simulation
                
                # Update ActualPosition
                updated_state.at[idx, 'ActualPosition'] = actual_pos
                
                # Update SeasonPoints (add points from this race)
                current_points = updated_state.at[idx, 'SeasonPoints']
                points_earned = points_system.get(actual_pos, 0)
                updated_state.at[idx, 'SeasonPoints'] = current_points + points_earned
                
                # Update SeasonAvgFinish (recalculate average)
                # Get all previous positions including this one
                current_avg = updated_state.at[idx, 'SeasonAvgFinish']
                if pd.isna(current_avg):
                    updated_state.at[idx, 'SeasonAvgFinish'] = float(actual_pos)
                else:
                    # Approximate: assume this is race N, update average
                    # This is simplified - ideally we'd track all previous positions
                    updated_state.at[idx, 'SeasonAvgFinish'] = (current_avg * 0.9 + actual_pos * 0.1)
                
                # Update RecentForm (last 5 races average)
                # Simplified: shift the average
                current_recent_form = updated_state.at[idx, 'RecentForm']
                if pd.isna(current_recent_form):
                    updated_state.at[idx, 'RecentForm'] = float(actual_pos)
                else:
                    # Approximate rolling average (simplified)
                    updated_state.at[idx, 'RecentForm'] = (current_recent_form * 0.8 + actual_pos * 0.2)
    
    # Update ConstructorPoints and ConstructorStanding based on team points
    # Group by constructor (simplified - would need TeamName column)
    # For now, we'll skip this as it requires constructor mapping
    
    return updated_state


def get_future_races(year: int):
    """
    Get future races (scheduled but not yet completed) from Fast F1 schedule.
    Uses cached schedule data without loading race sessions to avoid API calls.
    
    Args:
        year: Season year
        
    Returns:
        DataFrame with future race schedule information
    """
    try:
        # Lazy import to avoid triggering cache on every predict.py run
        import fastf1
        
        # Only get schedule - don't load sessions (that triggers API calls)
        schedule = fastf1.get_event_schedule(year)
        if schedule is None or schedule.empty:
            return pd.DataFrame()
        
        # Get all races from schedule
        # We'll determine if they're future races by checking if they exist in test data
        # This avoids loading sessions which triggers API calls
        all_races = []
        for _, event in schedule.iterrows():
            all_races.append({
                'Year': year,
                'EventName': event['EventName'],
                'RoundNumber': event['RoundNumber'],
                'Date': event.get('EventDate', ''),
                'Location': event.get('Location', '')
            })
        
        return pd.DataFrame(all_races)
    except Exception as e:
        # If fastf1 isn't available or fails, just return empty
        return pd.DataFrame()


def select_race_interactive(test_df: pd.DataFrame, training_df: pd.DataFrame = None):
    """
    Interactive function to let user select year and race from available options.
    Includes future races that haven't happened yet.
    
    Args:
        test_df: DataFrame with test data containing Year, EventName, RoundNumber columns
        training_df: Optional DataFrame with training data for historical track averages
        
    Returns:
        Tuple of (selected_df, input_source_string, is_future_race) or (None, None, False) if cancelled.
        If user types "all" for 2025, returns (list_of_race_tuples, "All 2025 races", 'all')
        where list_of_race_tuples is a list of (race_df, input_source, is_future) tuples.
    """
    if 'Year' not in test_df.columns or 'EventName' not in test_df.columns:
        return None, None, False
    
    # Get unique years from test data
    unique_years = sorted(test_df['Year'].unique())
    
    # Note: Future race checking is done lazily when user selects a year
    # to avoid triggering Fast F1 API calls on every predict.py run
    
    print("\n" + "=" * 70)
    print("Available Years:")
    print("=" * 70)
    for idx, year in enumerate(unique_years, 1):
        print(f"  {idx}. {year}")
    
    while True:
        try:
            year_choice = input(f"\nSelect year (1-{len(unique_years)}) or 'q' to quit: ").strip()
            if year_choice.lower() == 'q':
                return None, None, False
            
            year_idx = int(year_choice) - 1
            if 0 <= year_idx < len(unique_years):
                selected_year = unique_years[year_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(unique_years)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
    
    # Get races from test data (completed races)
    year_data = test_df[test_df['Year'] == selected_year]
    completed_races = year_data[['Year', 'EventName', 'RoundNumber']].drop_duplicates()
    
    # Get future races for this year
    future_races = get_future_races(selected_year)
    
    # Combine completed and future races
    if not future_races.empty:
        # Merge future races (avoid duplicates)
        future_races_clean = future_races[['Year', 'EventName', 'RoundNumber']].drop_duplicates()
        # Only add future races that aren't already in completed races
        future_only = future_races_clean[
            ~future_races_clean.set_index(['Year', 'EventName', 'RoundNumber']).index.isin(
                completed_races.set_index(['Year', 'EventName', 'RoundNumber']).index
            )
        ]
        all_races = pd.concat([completed_races, future_only], ignore_index=True)
    else:
        all_races = completed_races
    
    all_races = all_races.sort_values('RoundNumber')
    
    # Check which races are future (no data in test_df)
    is_future_list = []
    for _, race in all_races.iterrows():
        has_data = not test_df[
            (test_df['Year'] == race['Year']) & 
            (test_df['EventName'] == race['EventName']) &
            (test_df['RoundNumber'] == race['RoundNumber'])
        ].empty
        is_future_list.append(not has_data)
    
    print(f"\n" + "=" * 70)
    print(f"Available Races for {selected_year}:")
    print("=" * 70)
    for idx, (_, race) in enumerate(all_races.iterrows(), 1):
        future_marker = " [FUTURE]" if is_future_list[idx-1] else ""
        print(f"  {idx}. {race['EventName']} (Round {race['RoundNumber']}){future_marker}")
    
    while True:
        try:
            race_choice = input(f"\nSelect race (1-{len(all_races)}) or 'q' to quit" + (f" or 'all' to predict all {selected_year} races" if selected_year == 2025 else "") + ": ").strip()
            if race_choice.lower() == 'q':
                return None, None, False
            
            # Check for "all" option (only for 2025)
            if race_choice.lower() == 'all' and selected_year == 2025:
                # Process all races for 2025 and return list of race dataframes
                all_race_data = []
                print(f"\nProcessing all {len(all_races)} races for {selected_year}...")
                for idx, (_, race_row) in enumerate(all_races.iterrows(), 1):
                    race_name = race_row['EventName']
                    race_round = race_row['RoundNumber']
                    print(f"  [{idx}/{len(all_races)}] Processing {race_name} (Round {race_round})...")
                    
                    # Check if this is a future race (no data in test_df)
                    race_df = test_df[
                        (test_df['Year'] == race_row['Year']) & 
                        (test_df['EventName'] == race_row['EventName']) &
                        (test_df['RoundNumber'] == race_row['RoundNumber'])
                    ].copy()
                    
                    if race_df.empty:
                        # Future race - calculate features using most recent race data
                        try:
                            race_df = calculate_future_race_features(
                                test_df, selected_year, race_round, race_name, 
                                training_df if training_df is not None else None
                            )
                            input_source = f"{race_name} ({selected_year}, Round {race_round}) [FUTURE]"
                            all_race_data.append((race_df, input_source, True))
                        except Exception as e:
                            print(f"    Error calculating features for {race_name}: {e}")
                            print(f"    Skipping this race.")
                    else:
                        input_source = f"{race_name} ({selected_year}, Round {race_round})"
                        all_race_data.append((race_df, input_source, False))
                
                # Return special marker to indicate "all races" mode
                return all_race_data, f"All {selected_year} races", 'all'
            
            race_idx = int(race_choice) - 1
            if 0 <= race_idx < len(all_races):
                selected_race = all_races.iloc[race_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(all_races)}")
        except ValueError:
            if race_choice.lower() == 'all' and selected_year != 2025:
                print("'all' option is only available for 2025. Please select a specific race.")
            else:
                print("Please enter a valid number or 'q' to quit")
    
    # Check if this is a future race (no data in test_df)
    race_df = test_df[
        (test_df['Year'] == selected_race['Year']) & 
        (test_df['EventName'] == selected_race['EventName']) &
        (test_df['RoundNumber'] == selected_race['RoundNumber'])
    ].copy()
    
    if race_df.empty:
        # Future race - calculate features using most recent race data
        print(f"\n  This is a future race. Calculating features from most recent race data...")
        try:
            # Calculate features for future race using most recent completed race
            # Load training data if not already loaded
            if 'training_df' not in locals():
                script_dir = Path(__file__).parent
                training_data_path = script_dir.parent / 'data' / 'training_data.csv'
                training_df_local = None
                if training_data_path.exists():
                    try:
                        training_df_local = pd.read_csv(training_data_path)
                    except Exception:
                        pass
            else:
                training_df_local = training_df
            
            race_df = calculate_future_race_features(
                test_df, selected_year, selected_race['RoundNumber'], selected_race['EventName'], training_df_local
            )
            print(f"  Calculated features for {len(race_df)} drivers")
            print(f"  Note: GridPosition is unknown for future races (will use NaN/median)")
            
            input_source = f"{selected_race['EventName']} ({selected_year}, Round {selected_race['RoundNumber']}) [FUTURE]"
            return race_df, input_source, True
        except Exception as e:
            print(f"  Error calculating features for future race: {e}")
            print(f"  Please use --input-file with driver features for future races.")
            return None, None, False
    
    input_source = f"{selected_race['EventName']} ({selected_year}, Round {selected_race['RoundNumber']})"
    
    return race_df, input_source, False


def main():
    """Main function for making predictions."""
    parser = argparse.ArgumentParser(description='F1 Race Position Prediction (Top 10 Model) - Top 10')
    parser.add_argument('--input-file', type=str, help='CSV file with driver features for a race')
    parser.add_argument('--output-file', type=str, default='predictions.csv', help='Output file for predictions')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory containing model files')
    parser.add_argument('--model-type', type=str, default='neural_network', 
                       choices=['neural_network', 'random_forest'],
                       help='Type of model to use (neural_network or random_forest)')
    parser.add_argument('--show-all', action='store_true', help='Show all drivers, not just top 10')
    parser.add_argument('--race-name', type=str, help='Specific race name to predict (e.g., "Sao Paulo", "Brazil GP"). Default: Sao Paulo GP 2025. If not found, uses most recent race.')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode: select year and race from available options')
    
    args = parser.parse_args()
    
    print("F1 Race Position Prediction")
    print("=" * 70)
    print(f"Model Type: {args.model_type}")
    print("=" * 70)
    
    # Load model
    print("Loading model...")
    try:
        model, scaler, model_type, device = load_model(args.model_dir, args.model_type)
        print(f"Model loaded successfully! ({model_type})")
        print(f"Using device: {device}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nMake sure you've trained the model first:")
        if args.model_type == 'neural_network':
            print("  python train.py")
        else:
            print("  python train_rf.py")
        return
    
    # Make predictions
    if args.input_file:
        # Predict from provided file (expects one row per driver in the race)
        print(f"\nReading input from {args.input_file}...")
        df = pd.read_csv(args.input_file)
        print(f"  Found {len(df)} drivers")
        input_source = args.input_file
    else:
        # Try to use test data
        script_dir = Path(__file__).parent
        test_data_path = script_dir.parent / 'data' / 'test_data.csv'
        if not test_data_path.exists():
            print("\nError: No input file provided and test data not found.")
            print(f"  Expected test data at: {test_data_path}")
            print("\nPlease provide --input-file with driver features for the race.")
            print("\nThe input CSV should have one row per driver with columns:")
            print("  - SeasonPoints")
            print("  - SeasonAvgFinish")
            print("  - HistoricalTrackAvgPosition")
            print("  - ConstructorStanding")
            print("  - GridPosition")
            print("  - RecentForm")
            print("  - TrackType")
            print("  - DriverNumber (optional)")
            print("  - DriverName (optional)")
            print("\nExample:")
            print("  python predict.py --input-file race_drivers.csv")
            print("\nOr run collect_data.py first to generate test data.")
            return
        
        test_df = pd.read_csv(test_data_path)
        
        # Load training data for historical track averages (if available)
        training_data_path = Path('data') / 'training_data.csv'
        training_df = None
        if training_data_path.exists():
            try:
                training_df = pd.read_csv(training_data_path)
            except Exception:
                pass  # If training data can't be loaded, continue without it
        
        # Interactive mode: let user select year and race (default when no input file and no race-name specified)
        # Only skip interactive if --race-name is explicitly provided
        if args.race_name:
            # Non-interactive: use race-name to find race
            print(f"\nNo input file provided. Using test data from {test_data_path}...")
            
            # Group by race (Year, EventName, RoundNumber) and use the most recent/upcoming race
            if 'Year' in test_df.columns and 'EventName' in test_df.columns and 'RoundNumber' in test_df.columns:
                # Get unique races
                unique_races = test_df[['Year', 'EventName', 'RoundNumber']].drop_duplicates()
                if len(unique_races) > 0:
                    # Sort by RoundNumber descending to get the most recent/upcoming race
                    unique_races = unique_races.sort_values('RoundNumber', ascending=False)
                    
                    # Look for the specified race name
                    matching_races = unique_races[unique_races['EventName'].str.contains(args.race_name, case=False, na=False)]
                    if len(matching_races) > 0:
                        selected_race = matching_races.iloc[0]
                        print(f"  Found specified race: {args.race_name}")
                    else:
                        # Try to find Sao Paulo GP (default for 2025 prediction)
                        sao_paulo_race = unique_races[unique_races['EventName'].str.contains('Sao Paulo|São Paulo|Brazil|Brazilian', case=False, na=False)]
                        if len(sao_paulo_race) > 0:
                            selected_race = sao_paulo_race.iloc[0]
                            print(f"  Found Sao Paulo GP in test data!")
                        else:
                            # Use the most recent race (highest round number)
                            selected_race = unique_races.iloc[0]
                            print(f"  Using most recent/upcoming race from test data...")
                    
                    race_year = selected_race['Year']
                    race_event = selected_race['EventName']
                    race_round = selected_race['RoundNumber']
                    
                    # Filter to just this race
                    df = test_df[
                        (test_df['Year'] == race_year) & 
                        (test_df['EventName'] == race_event) &
                        (test_df['RoundNumber'] == race_round)
                    ].copy()
                    print(f"  Race: {race_event} ({race_year}, Round {race_round})")
                    print(f"  Found {len(df)} drivers")
                    input_source = f"test data ({race_event})"
                else:
                    # Fallback: just use all test data
                    df = test_df.copy()
                    print(f"  Found {len(df)} drivers (all test data)")
                    input_source = "test data"
            else:
                # No grouping columns, use all test data
                df = test_df.copy()
                print(f"  Found {len(df)} drivers")
                input_source = "test data"
        else:
            # Interactive mode: let user select year and race
            print(f"\nLoading test data from {test_data_path}...")
            result = select_race_interactive(test_df, training_df)
            if result[0] is None:
                print("\nSelection cancelled. Exiting.")
                return
            
            df, input_source, is_future_race = result
            
            # Check if "all" mode was selected (returns list of races)
            if is_future_race == 'all':
                # Process all races
                all_race_data = df  # df is actually the list of (race_df, input_source, is_future) tuples
                print(f"\n{'='*70}")
                print(f"PREDICTING ALL {len(all_race_data)} RACES FOR {input_source}")
                print(f"{'='*70}\n")
                
                all_predictions = []
                # Track current state for progressive feature updates across future races
                current_state_df = None
                # Load training data for historical track averages
                script_dir = Path(__file__).parent
                training_data_path = script_dir.parent / 'data' / 'training_data.csv'
                training_df_all = None
                if training_data_path.exists():
                    try:
                        training_df_all = pd.read_csv(training_data_path)
                    except Exception:
                        pass
                
                for race_idx, (race_df, race_input_source, is_future) in enumerate(all_race_data, 1):
                    print(f"\n{'='*70}")
                    print(f"Race {race_idx}/{len(all_race_data)}: {race_input_source}")
                    print(f"{'='*70}")
                    print(f"  Found {len(race_df)} drivers")
                    
                    # For future races, progressively update features based on previous race
                    if is_future:
                        if current_state_df is not None:
                            # For subsequent future races, recalculate ALL features from the updated state
                            # Get track name from race_input_source or race_df
                            track_name = race_df['EventName'].iloc[0] if 'EventName' in race_df.columns else race_input_source.split('(')[0].strip()
                            
                            # Recalculate features using the updated state from previous future race
                            # This ensures features reflect all previous races including simulated future ones
                            race_df = recalculate_features_from_state(
                                race_df, current_state_df, track_name, training_df_all
                            )
                        else:
                            # First future race - just ensure track-specific averages are correct
                            track_name = race_df['EventName'].iloc[0] if 'EventName' in race_df.columns else race_input_source.split('(')[0].strip()
                            if training_df_all is not None:
                                # Recalculate track-specific averages
                                track_historical = training_df_all[training_df_all['EventName'] == track_name]
                                if not track_historical.empty:
                                    for idx, row in race_df.iterrows():
                                        driver_num = row['DriverNumber']
                                        driver_track_races = track_historical[track_historical['DriverNumber'] == driver_num]
                                        if 'ActualPosition' in driver_track_races.columns:
                                            valid_positions = driver_track_races['ActualPosition'].dropna()
                                            if len(valid_positions) > 0:
                                                race_df.at[idx, 'HistoricalTrackAvgPosition'] = valid_positions.mean()
                    
                    try:
                        top10, all_results = predict_race_top10(race_df, model, scaler, model_type, device)
                        
                        # Check if this is a future race (no actual positions available)
                        race_is_future_check = 'ActualPosition' not in all_results.columns or all_results['ActualPosition'].isna().all()
                        
                        # Display ranking accuracy analysis table (for "all" mode)
                        if not race_is_future_check and 'ActualPosition' in all_results.columns:
                            print(f"\n{'='*70}")
                            print(f"RANKING ACCURACY ANALYSIS")
                            print(f"{'='*70}")
                            print(f"Comparing: Predicted Rank (1-10) vs Actual Finishing Position (1-20)")
                            print()
                            print(f"{'Driver':<12} {'Pred':<6} {'Actual':<8} {'GridPos':<8} {'Error':<7} {'Status':<8} {'SeasPts':<8} {'SeasAvg':<8} {'TrackAvg':<9} {'ConstrSt':<9} {'Form':<7} {'TrackType':<9}")
                            print("-" * 120)
                            
                            for _, row in top10.iterrows():
                                if not pd.isna(row.get('ActualPosition')):
                                    pred_rank = row['Rank']
                                    actual_pos = int(row['ActualPosition'])
                                    grid_pos = row.get('GridPosition', 'N/A')
                                    if pd.isna(grid_pos):
                                        grid_pos = 'N/A'
                                    else:
                                        grid_pos = int(grid_pos)
                                    error = abs(pred_rank - actual_pos)
                                    driver_name = row.get('DriverName', f"Driver {row['DriverNumber']}")
                                    status = get_status(error)
                                    
                                    # Get only features used by model (7 features)
                                    season_pts = row.get('SeasonPoints', 0)
                                    season_avg = row.get('SeasonAvgFinish', 0)
                                    track_avg = row.get('HistoricalTrackAvgPosition', 10.0)
                                    if pd.isna(track_avg):
                                        track_avg = 10.0
                                    constr_st = row.get('ConstructorStanding', 0)
                                    recent_form = row.get('RecentForm', 0)
                                    track_type = row.get('TrackType', 0)
                                    
                                    # Format values
                                    season_pts_str = f"{season_pts:.0f}" if not pd.isna(season_pts) else "N/A"
                                    season_avg_str = f"{season_avg:.2f}" if not pd.isna(season_avg) else "N/A"
                                    track_avg_str = f"{track_avg:.2f}"  # Always show a number (default 10.0 for rookies)
                                    constr_st_str = f"{constr_st:.0f}" if not pd.isna(constr_st) else "N/A"
                                    recent_form_str = f"{recent_form:.2f}" if not pd.isna(recent_form) else "N/A"
                                    track_type_str = "Street" if track_type == 1 else "Permanent"
                                    
                                    print(f"{driver_name:<12} {pred_rank:<6} {actual_pos:<8} {grid_pos:<8} {error:<7} {status:<8} {season_pts_str:<8} {season_avg_str:<8} {track_avg_str:<9} {constr_st_str:<9} {recent_form_str:<7} {track_type_str:<9}")
                            
                            print("-" * 150)
                        else:
                            # Future race - show table with all features
                            print(f"\nPREDICTED TOP 10 FINISHERS")
                            print("-" * 120)
                            print(f"{'Rank':<6} {'Driver':<12} {'Pred':<6} {'SeasPts':<8} {'SeasAvg':<8} {'TrackAvg':<9} {'ConstrSt':<9} {'GridPos':<8} {'Form':<7} {'TrackType':<9}")
                            print("-" * 120)
                            
                            for _, row in top10.iterrows():
                                driver_name = row.get('DriverName', f"Driver {row.get('DriverNumber', 'N/A')}")
                                rank = row['Rank']
                                pred_pos = row['PredictedPosition']
                                season_pts = row.get('SeasonPoints', 0)
                                season_avg = row.get('SeasonAvgFinish', 0)
                                track_avg = row.get('HistoricalTrackAvgPosition', 10.0)
                                if pd.isna(track_avg):
                                    track_avg = 10.0
                                constr_st = row.get('ConstructorStanding', 0)
                                grid_pos = row.get('GridPosition', 0)
                                recent_form = row.get('RecentForm', 0)
                                track_type = row.get('TrackType', 0)
                                
                                # Format values
                                season_pts_str = f"{season_pts:.0f}" if not pd.isna(season_pts) else "N/A"
                                season_avg_str = f"{season_avg:.2f}" if not pd.isna(season_avg) else "N/A"
                                track_avg_str = f"{track_avg:.2f}"  # Always show a number (default 10.0 for rookies)
                                constr_st_str = f"{constr_st:.0f}" if not pd.isna(constr_st) else "N/A"
                                grid_pos_str = f"{grid_pos:.0f}" if not pd.isna(grid_pos) else "N/A"
                                recent_form_str = f"{recent_form:.2f}" if not pd.isna(recent_form) else "N/A"
                                track_type_str = "Street" if track_type == 1 else "Permanent"
                                
                                print(f"{rank:<6} {driver_name:<12} {pred_pos:<6.3f} {season_pts_str:<8} {season_avg_str:<8} {track_avg_str:<9} {constr_st_str:<9} {grid_pos_str:<8} {recent_form_str:<7} {track_type_str:<9}")
                        
                        # Store predictions
                        top10_copy = top10.copy()
                        top10_copy['Race'] = race_input_source
                        all_predictions.append(top10_copy)
                        
                        # Update current state for next future race (use predictions as simulated results)
                        if is_future:
                            if current_state_df is None:
                                current_state_df = race_df.copy()
                            current_state_df = update_state_with_predictions(current_state_df, all_results)
                        
                        # Show summary statistics
                        print(f"\nSummary:")
                        print(f"  Best predicted: Position {top10['PredictedPosition'].min():.2f}")
                        print(f"  Worst in top 10: Position {top10['PredictedPosition'].max():.2f}")
                        print(f"  Average predicted position (top 10): {top10['PredictedPosition'].mean():.2f}")
                        
                        # If actual positions are available, show ranking accuracy (skip for future races)
                        if not race_is_future_check and 'ActualPosition' in all_results.columns:
                            # Calculate ranking accuracy
                            exact_matches = sum(1 for _, row in top10.iterrows() 
                                              if 'ActualPosition' in row and not pd.isna(row['ActualPosition']) 
                                              and abs(row['Rank'] - row['ActualPosition']) == 0)
                            within_1 = sum(1 for _, row in top10.iterrows() 
                                         if 'ActualPosition' in row and not pd.isna(row['ActualPosition']) 
                                         and abs(row['Rank'] - row['ActualPosition']) <= 1)
                            within_2 = sum(1 for _, row in top10.iterrows() 
                                         if 'ActualPosition' in row and not pd.isna(row['ActualPosition']) 
                                         and abs(row['Rank'] - row['ActualPosition']) <= 2)
                            within_3 = sum(1 for _, row in top10.iterrows() 
                                         if 'ActualPosition' in row and not pd.isna(row['ActualPosition']) 
                                         and abs(row['Rank'] - row['ActualPosition']) <= 3)
                            
                            print(f"\nRanking Accuracy:")
                            print(f"  Exact matches: {exact_matches}/10 ({exact_matches*10:.1f}%)")
                            print(f"  Within 1 position: {within_1}/10 ({within_1*10:.1f}%)")
                            print(f"  Within 2 positions: {within_2}/10 ({within_2*10:.1f}%)")
                            print(f"  Within 3 positions: {within_3}/10 ({within_3*10:.1f}%)")
                        
                    except Exception as e:
                        print(f"  Error predicting race {race_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # Save all predictions
                if all_predictions:
                    combined_predictions = pd.concat(all_predictions, ignore_index=True)
                    output_path = Path(args.output_file)
                    combined_predictions.to_csv(output_path, index=False)
                    print(f"\n{'='*70}")
                    print(f"All predictions saved to {output_path}")
                    print(f"Total races predicted: {len(all_predictions)}")
                    print(f"{'='*70}")
                
                return
            
            # Single race mode (normal flow)
            if is_future_race:
                print(f"\nSelected race: {input_source}")
                print(f"  This is a future race - predictions will be made but accuracy cannot be calculated.")
                print(f"  Features calculated from most recent completed race.")
                print(f"  Found {len(df)} drivers")
            else:
                print(f"\nSelected race: {input_source}")
                print(f"  Found {len(df)} drivers")
    
    # Predict positions and get top 10
    try:
        top10, all_results = predict_race_top10(df, model, scaler, model_type, device)
        
        # Check if this is a future race (no actual positions available)
        is_future_race = 'ActualPosition' not in all_results.columns or all_results['ActualPosition'].isna().all()
        
        # Display top 10
        print("\n" + "=" * 70)
        print(f"PREDICTED TOP 10 FINISHERS ({input_source})")
        print("=" * 70)
        if not is_future_race and 'ActualPosition' in all_results.columns:
            print(f"{'Rank':<6} {'Driver':<20} {'Driver #':<10} {'Predicted':<12} {'Actual':<10} {'AvgGrid':<10} {'Status':<10}")
        else:
            # Future race - show only features used by model (7 features)
            print(f"{'Rank':<6} {'Driver':<12} {'Pred':<6} {'SeasPts':<8} {'SeasAvg':<8} {'TrackAvg':<9} {'ConstrSt':<9} {'GridPos':<8} {'Form':<7} {'TrackType':<9}")
        print("-" * 120 if is_future_race else "-" * 70)
        
        for _, row in top10.iterrows():
            driver_name = row.get('DriverName', f"Driver {row.get('DriverNumber', 'N/A')}")
            driver_num = row.get('DriverNumber', 'N/A')
            pred_pos = row['PredictedPosition']
            grid_pos = row.get('GridPosition', 'N/A')
            rank = row['Rank']
            
            if not is_future_race and 'ActualPosition' in row and not pd.isna(row['ActualPosition']):
                actual_pos = int(row['ActualPosition'])
                status = get_status(abs(rank - actual_pos))
                print(f"{rank:<6} {driver_name:<20} {driver_num:<10} {pred_pos:<12.3f} {actual_pos:<10} {grid_pos:<10} {status:<10}")
            else:
                # Future race - show only features used by model (7 features)
                season_pts = row.get('SeasonPoints', 0)
                season_avg = row.get('SeasonAvgFinish', 0)
                track_avg = row.get('HistoricalTrackAvgPosition', 10.0)
                if pd.isna(track_avg):
                    track_avg = 10.0
                constr_st = row.get('ConstructorStanding', 0)
                grid_pos_val = row.get('GridPosition', 0)
                recent_form = row.get('RecentForm', 0)
                track_type = row.get('TrackType', 0)
                
                # Format values
                season_pts_str = f"{season_pts:.0f}" if not pd.isna(season_pts) else "N/A"
                season_avg_str = f"{season_avg:.2f}" if not pd.isna(season_avg) else "N/A"
                track_avg_str = f"{track_avg:.2f}"  # Always show a number (default 10.0 for rookies)
                constr_st_str = f"{constr_st:.0f}" if not pd.isna(constr_st) else "N/A"
                grid_pos_str = f"{grid_pos_val:.0f}" if not pd.isna(grid_pos_val) else "N/A"
                recent_form_str = f"{recent_form:.2f}" if not pd.isna(recent_form) else "N/A"
                track_type_str = "Street" if track_type == 1 else "Permanent"
                
                print(f"{rank:<6} {driver_name:<12} {pred_pos:<6.3f} {season_pts_str:<8} {season_avg_str:<8} {track_avg_str:<9} {constr_st_str:<9} {grid_pos_str:<8} {recent_form_str:<7} {track_type_str:<9}")
        
        # Save predictions
        output_path = Path(args.output_file)
        if args.show_all:
            all_results.to_csv(output_path, index=False)
            print(f"\nAll {len(all_results)} driver predictions saved to {output_path}")
        else:
            top10.to_csv(output_path, index=False)
            print(f"\nTop 5 predictions saved to {output_path}")
        
        # Show summary statistics
        print(f"\nSummary:")
        print(f"  Best predicted: Position {top10['PredictedPosition'].min():.2f}")
        print(f"  Worst in top 10: Position {top10['PredictedPosition'].max():.2f}")
        print(f"  Average predicted position (top 10): {top10['PredictedPosition'].mean():.2f}")
        
        # If actual positions are available, show ranking accuracy (skip for future races)
        if not is_future_race and 'ActualPosition' in all_results.columns:
            # Calculate ranking accuracy - how well does predicted rank match actual position?
            exact_matches = 0
            within_1 = 0
            within_2 = 0
            within_3 = 0
            ranking_errors = []
            
            print(f"\n" + "=" * 70)
            print(f"RANKING ACCURACY ANALYSIS")
            print("=" * 70)
            print(f"Comparing: Predicted Rank (1-10) vs Actual Finishing Position (1-20)")
            print()
            
            # Show detailed comparison table
            print(f"{'Driver':<12} {'Pred':<6} {'Actual':<8} {'GridPos':<8} {'Error':<7} {'Status':<8} {'SeasPts':<8} {'SeasAvg':<8} {'TrackAvg':<9} {'ConstrSt':<9} {'Form':<7} {'TrackType':<9}")
            print("-" * 120)
            
            for _, row in top10.iterrows():
                if not pd.isna(row.get('ActualPosition')):
                    pred_rank = row['Rank']
                    actual_pos = int(row['ActualPosition'])
                    grid_pos = row.get('GridPosition', 'N/A')
                    if pd.isna(grid_pos):
                        grid_pos = 'N/A'
                    else:
                        grid_pos = int(grid_pos)
                    error = abs(pred_rank - actual_pos)
                    driver_name = row.get('DriverName', f"Driver {row['DriverNumber']}")
                    ranking_errors.append((driver_name, pred_rank, actual_pos, error))
                    
                    status = get_status(error)
                    if error == 0:
                        exact_matches += 1
                    if error <= 1:
                        within_1 += 1
                    if error <= 2:
                        within_2 += 1
                    if error <= 3:
                        within_3 += 1
                    
                    # Get only features used by model (7 features)
                    season_pts = row.get('SeasonPoints', 0)
                    season_avg = row.get('SeasonAvgFinish', 0)
                    track_avg = row.get('HistoricalTrackAvgPosition', 10.0)
                    if pd.isna(track_avg):
                        track_avg = 10.0
                    constr_st = row.get('ConstructorStanding', 0)
                    recent_form = row.get('RecentForm', 0)
                    track_type = row.get('TrackType', 0)
                    
                    # Format values
                    season_pts_str = f"{season_pts:.0f}" if not pd.isna(season_pts) else "N/A"
                    season_avg_str = f"{season_avg:.2f}" if not pd.isna(season_avg) else "N/A"
                    track_avg_str = f"{track_avg:.2f}"  # Always show a number (default 10.0 for rookies)
                    constr_st_str = f"{constr_st:.0f}" if not pd.isna(constr_st) else "N/A"
                    recent_form_str = f"{recent_form:.2f}" if not pd.isna(recent_form) else "N/A"
                    track_type_str = "Street" if track_type == 1 else "Permanent"
                    
                    print(f"{driver_name:<12} {pred_rank:<6} {actual_pos:<8} {grid_pos:<8} {error:<7} {status:<8} {season_pts_str:<8} {season_avg_str:<8} {track_avg_str:<9} {constr_st_str:<9} {recent_form_str:<7} {track_type_str:<9}")
            
            print("-" * 120)
            print(f"\nSummary:")
            print(f"  Exact matches (error = 0): {exact_matches}/10 ({exact_matches/10*100:.1f}%)")
            print(f"  Within 1 position: {within_1}/10 ({within_1/10*100:.1f}%)")
            print(f"  Within 2 positions: {within_2}/10 ({within_2/10*100:.1f}%)")
            print(f"  Within 3 positions: {within_3}/10 ({within_3/10*100:.1f}%)")
            
            if ranking_errors:
                avg_error = sum(e[3] for e in ranking_errors) / len(ranking_errors)
                print(f"\n  Average ranking error: {avg_error:.2f} positions")
                
                # Show drivers with largest ranking errors
                ranking_errors.sort(key=lambda x: x[3], reverse=True)
                print(f"\n  Worst predictions (largest errors):")
                for driver_name, pred_rank, actual_pos, error in ranking_errors[:3]:
                    print(f"    - {driver_name}: Predicted rank {pred_rank}, Actual position {actual_pos} (error: {error} positions)")
            
            # Calculate Spearman rank correlation for top 10
            try:
                from scipy.stats import spearmanr
                pred_ranks = [row['Rank'] for _, row in top10.iterrows() if not pd.isna(row.get('ActualPosition'))]
                actual_positions = [int(row['ActualPosition']) for _, row in top10.iterrows() if not pd.isna(row.get('ActualPosition'))]
                
                if len(pred_ranks) > 1:
                    correlation, _ = spearmanr(pred_ranks, actual_positions)
                    print(f"\n  Rank correlation (Spearman): {correlation:.3f}")
                    if correlation > 0.7:
                        print(f"    → Strong positive correlation (good ranking order)")
                    elif correlation > 0.4:
                        print(f"    → Moderate positive correlation")
                    elif correlation > 0:
                        print(f"    → Weak positive correlation")
                    else:
                        print(f"    → Poor/no correlation")
            except ImportError:
                pass  # scipy not available, skip correlation
            
    except Exception as e:
        print(f"\nError making predictions: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
