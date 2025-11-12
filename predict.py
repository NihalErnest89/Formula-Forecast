"""
Inference script for F1 Predictions.
Loads trained model (Random Forest or Neural Network) and makes predictions for race positions.
Ranks drivers and displays predicted top 10.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import torch
import torch.nn as nn


class F1NeuralNetwork(nn.Module):
    """Neural Network model definition (must match training - regression)."""
    
    def __init__(self, input_size=6, hidden_sizes=[128, 64, 32], dropout_rate=0.3):
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
        return self.network(x).squeeze()


def load_model(model_dir: str = 'models', model_type: str = 'neural_network', auto_fallback: bool = True):
    """
    Load trained model and scaler.
    
    Args:
        model_dir: Directory containing model files
        model_type: 'neural_network' or 'random_forest'
        auto_fallback: If True, try the other model type if requested one is not found
        
    Returns:
        Tuple of (model, scaler, model_type, device)
    """
    # Try to load the requested model type
    requested_type = model_type
    tried_nn = False
    
    if model_type == 'neural_network':
        nn_model_path = Path(model_dir) / 'f1_predictor_model.pth'
        nn_scaler_path = Path(model_dir) / 'scaler.pkl'
        tried_nn = True
        
        if nn_model_path.exists() and nn_scaler_path.exists():
            # Initialize and load model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = F1NeuralNetwork(
                input_size=6,
                hidden_sizes=[128, 64, 32],
                dropout_rate=0.3
            ).to(device)
            model.load_state_dict(torch.load(nn_model_path, map_location=device))
            model.eval()
            
            with open(nn_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            return model, scaler, 'neural_network', device
        elif auto_fallback:
            # Try Random Forest as fallback
            print(f"Warning: Neural network model not found at {nn_model_path}")
            print("Attempting to load Random Forest model instead...")
            model_type = 'random_forest'  # Switch to try RF
        else:
            raise FileNotFoundError(f"Model not found at {nn_model_path}. Run train.py first.")
    
    # Try Random Forest (either requested or as fallback)
    if model_type == 'random_forest':
        rf_model_path = Path(model_dir) / 'f1_predictor_model_rf.pkl'
        rf_scaler_path = Path(model_dir) / 'scaler_rf.pkl'
        
        if rf_model_path.exists() and rf_scaler_path.exists():
            with open(rf_model_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(rf_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            return model, scaler, 'random_forest', None
        elif tried_nn and auto_fallback:
            # Tried both, neither found
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
    
    raise ValueError(f"Unknown model type: {requested_type}")


def predict_position(season_points: float, season_avg_finish: float, 
                    historical_track_avg: float, constructor_points: float,
                    constructor_standing: float, grid_position: float,
                    model, scaler, model_type='neural_network', device=None):
    """
    Predict finishing position for a driver given input features.
    
    Args:
        season_points: Season points accumulated so far
        season_avg_finish: Average finish position this season
        historical_track_avg: Historical average position at this track
        constructor_points: Constructor's total points
        constructor_standing: Constructor's championship standing
        grid_position: Starting grid position (qualifying)
        model: Trained model
        scaler: Fitted scaler
        model_type: 'neural_network' or 'random_forest'
        device: Device for neural network inference
        
    Returns:
        Predicted finishing position (1-20)
    """
    # Create feature vector
    features = np.array([[season_points, season_avg_finish, historical_track_avg,
                          constructor_points, constructor_standing, grid_position]])
    
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
    
    # Ensure position is within valid range (1-20)
    predicted_position = max(1, min(20, predicted_position))
    
    return predicted_position


def predict_race_top10(drivers_df: pd.DataFrame, model, scaler, 
                       model_type='neural_network', device=None):
    """
    Predict positions for all drivers in a race and return top 10.
    
    Args:
        drivers_df: DataFrame with one row per driver, containing:
                   ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                    'ConstructorPoints', 'ConstructorStanding', 'GridPosition', 'DriverNumber', 'DriverName']
        model: Trained model
        scaler: Fitted scaler
        model_type: 'neural_network' or 'random_forest'
        device: Device for neural network inference
        
    Returns:
        DataFrame with predicted positions, sorted to show top 10
    """
    # All 6 features (new version)
    all_feature_cols = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                        'ConstructorPoints', 'ConstructorStanding', 'GridPosition']
    
    # Check how many features the scaler expects
    expected_features = getattr(scaler, 'n_features_in_', None)
    
    # Determine which features to use based on what the model expects
    if expected_features == 3:
        # Old model trained with 3 features
        feature_cols = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition']
        print(f"Warning: Model was trained with 3 features. Using only: {feature_cols}")
        print("  To use all 6 features, please retrain the model with: python train_rf.py")
    elif expected_features == 6:
        # New model trained with 6 features
        feature_cols = all_feature_cols
    else:
        # Unknown, try to infer from scaler
        if expected_features is not None:
            if expected_features == 3:
                feature_cols = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition']
            else:
                feature_cols = all_feature_cols
        else:
            # Fallback: try 6 features first, if it fails, try 3
            feature_cols = all_feature_cols
    
    # Check if required columns exist
    missing_cols = [col for col in feature_cols if col not in drivers_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Prepare features
    X = drivers_df[feature_cols].values
    
    # Handle missing values - use median if mean fails (handles empty slices)
    with np.errstate(all='ignore'):
        # Try mean first
        col_means = np.nanmean(X, axis=0)
        # If any column has all NaN, use median or 0
        for i in range(X.shape[1]):
            if np.isnan(col_means[i]) or np.isinf(col_means[i]):
                median_val = np.nanmedian(X[:, i])
                if np.isnan(median_val):
                    fill_val = 0
                else:
                    fill_val = median_val
                X[:, i] = np.nan_to_num(X[:, i], nan=fill_val)
            else:
                X[:, i] = np.nan_to_num(X[:, i], nan=col_means[i])
    
    # Scale features - catch feature mismatch error
    try:
        X_scaled = scaler.transform(X)
    except ValueError as e:
        if "features" in str(e).lower():
            # Feature mismatch - try with 3 features if we were using 6
            if len(feature_cols) == 6 and expected_features != 6:
                print(f"Error: Model expects {expected_features} features but got {len(feature_cols)}.")
                print("Attempting with 3 features (old model format)...")
                feature_cols = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition']
                X = drivers_df[feature_cols].values
                X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
                X_scaled = scaler.transform(X)
            else:
                raise ValueError(
                    f"Feature mismatch: Model expects {expected_features} features, "
                    f"but data has {len(feature_cols)} features. "
                    f"Please retrain the model with the correct number of features."
                ) from e
        else:
            raise
    
    # Make predictions
    if model_type == 'neural_network':
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            predicted_positions = model(X_tensor).cpu().numpy()
    else:
        predicted_positions = model.predict(X_scaled)
    
    # Ensure positions are in valid range
    predicted_positions = np.clip(predicted_positions, 1, 20)
    
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
    # Use the same feature detection logic as predict_race_top10
    all_feature_cols = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                        'ConstructorPoints', 'ConstructorStanding', 'GridPosition']
    
    # Check how many features the scaler expects
    expected_features = getattr(scaler, 'n_features_in_', None)
    
    # Determine which features to use
    if expected_features == 3:
        feature_cols = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition']
    elif expected_features == 6:
        feature_cols = all_feature_cols
    else:
        feature_cols = all_feature_cols  # Default to 6
    
    # Check if required columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Prepare features
    X = df[feature_cols].values
    
    # Handle missing values - same logic as predict_race_top10
    with np.errstate(all='ignore'):
        col_means = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            if np.isnan(col_means[i]) or np.isinf(col_means[i]):
                median_val = np.nanmedian(X[:, i])
                fill_val = 0 if np.isnan(median_val) else median_val
                X[:, i] = np.nan_to_num(X[:, i], nan=fill_val)
            else:
                X[:, i] = np.nan_to_num(X[:, i], nan=col_means[i])
    
    # Scale features - catch feature mismatch
    try:
        X_scaled = scaler.transform(X)
    except ValueError as e:
        if "features" in str(e).lower() and len(feature_cols) == 6 and expected_features == 3:
            # Fallback to 3 features
            feature_cols = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition']
            X = df[feature_cols].values
            with np.errstate(all='ignore'):
                col_means = np.nanmean(X, axis=0)
                for i in range(X.shape[1]):
                    if np.isnan(col_means[i]) or np.isinf(col_means[i]):
                        median_val = np.nanmedian(X[:, i])
                        fill_val = 0 if np.isnan(median_val) else median_val
                        X[:, i] = np.nan_to_num(X[:, i], nan=fill_val)
                    else:
                        X[:, i] = np.nan_to_num(X[:, i], nan=col_means[i])
            X_scaled = scaler.transform(X)
        else:
            raise
    
    # Make predictions
    if model_type == 'neural_network':
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            predicted_positions = model(X_tensor).cpu().numpy()
    else:
        predicted_positions = model.predict(X_scaled)
    
    # Ensure positions are in valid range
    predicted_positions = np.clip(predicted_positions, 1, 20)
    
    # Add predictions to DataFrame
    df['PredictedPosition'] = predicted_positions
    
    return df


def select_race_interactive(test_df: pd.DataFrame):
    """
    Interactive function to let user select year and race from available options.
    
    Args:
        test_df: DataFrame with test data containing Year, EventName, RoundNumber columns
        
    Returns:
        Tuple of (selected_df, input_source_string) or (None, None) if cancelled
    """
    if 'Year' not in test_df.columns or 'EventName' not in test_df.columns:
        return None, None
    
    # Get unique years
    unique_years = sorted(test_df['Year'].unique())
    
    print("\n" + "=" * 70)
    print("Available Years:")
    print("=" * 70)
    for idx, year in enumerate(unique_years, 1):
        print(f"  {idx}. {year}")
    
    while True:
        try:
            year_choice = input(f"\nSelect year (1-{len(unique_years)}) or 'q' to quit: ").strip()
            if year_choice.lower() == 'q':
                return None, None
            
            year_idx = int(year_choice) - 1
            if 0 <= year_idx < len(unique_years):
                selected_year = unique_years[year_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(unique_years)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
    
    # Filter to selected year and get unique races
    year_data = test_df[test_df['Year'] == selected_year]
    unique_races = year_data[['Year', 'EventName', 'RoundNumber']].drop_duplicates()
    unique_races = unique_races.sort_values('RoundNumber')
    
    print(f"\n" + "=" * 70)
    print(f"Available Races for {selected_year}:")
    print("=" * 70)
    for idx, (_, race) in enumerate(unique_races.iterrows(), 1):
        print(f"  {idx}. {race['EventName']} (Round {race['RoundNumber']})")
    
    while True:
        try:
            race_choice = input(f"\nSelect race (1-{len(unique_races)}) or 'q' to quit: ").strip()
            if race_choice.lower() == 'q':
                return None, None
            
            race_idx = int(race_choice) - 1
            if 0 <= race_idx < len(unique_races):
                selected_race = unique_races.iloc[race_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(unique_races)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
    
    # Filter to selected race
    race_df = test_df[
        (test_df['Year'] == selected_race['Year']) & 
        (test_df['EventName'] == selected_race['EventName']) &
        (test_df['RoundNumber'] == selected_race['RoundNumber'])
    ].copy()
    
    input_source = f"{selected_race['EventName']} ({selected_year}, Round {selected_race['RoundNumber']})"
    
    return race_df, input_source


def main():
    """Main function for making predictions."""
    parser = argparse.ArgumentParser(description='F1 Race Position Prediction - Top 10')
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
        if device and device.type == 'cuda':
            print(f"Using GPU: {device}")
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
        test_data_path = Path('data') / 'test_data.csv'
        if not test_data_path.exists():
            print("\nError: No input file provided and test data not found.")
            print(f"  Expected test data at: {test_data_path}")
            print("\nPlease provide --input-file with driver features for the race.")
            print("\nThe input CSV should have one row per driver with columns:")
            print("  - SeasonPoints")
            print("  - SeasonAvgFinish")
            print("  - HistoricalTrackAvgPosition")
            print("  - ConstructorPoints")
            print("  - ConstructorStanding")
            print("  - GridPosition")
            print("  - DriverNumber (optional)")
            print("  - DriverName (optional)")
            print("\nExample:")
            print("  python predict.py --input-file race_drivers.csv")
            print("\nOr run collect_data.py first to generate test data.")
            return
        
        test_df = pd.read_csv(test_data_path)
        
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
            df, input_source = select_race_interactive(test_df)
            if df is None:
                print("\nSelection cancelled. Exiting.")
                return
            print(f"\nSelected race: {input_source}")
            print(f"  Found {len(df)} drivers")
    
    # Predict positions and get top 10
    try:
        top10, all_results = predict_race_top10(df, model, scaler, model_type, device)
        
        # Display top 10
        print("\n" + "=" * 70)
        print(f"PREDICTED TOP 10 FINISHERS ({input_source})")
        print("=" * 70)
        if 'ActualPosition' in all_results.columns:
            print(f"{'Rank':<6} {'Driver':<20} {'Driver #':<10} {'Predicted':<12} {'Actual':<10} {'Grid':<10}")
        else:
            print(f"{'Rank':<6} {'Driver':<20} {'Driver #':<10} {'Predicted Pos':<15} {'Grid Pos':<10}")
        print("-" * 70)
        
        for _, row in top10.iterrows():
            driver_name = row.get('DriverName', f"Driver {row.get('DriverNumber', 'N/A')}")
            driver_num = row.get('DriverNumber', 'N/A')
            pred_pos = row['PredictedPosition']
            grid_pos = row.get('GridPosition', 'N/A')
            rank = row['Rank']
            
            if 'ActualPosition' in row and not pd.isna(row['ActualPosition']):
                actual_pos = int(row['ActualPosition'])
                print(f"{rank:<6} {driver_name:<20} {driver_num:<10} {pred_pos:<12.2f} {actual_pos:<10} {grid_pos:<10}")
            else:
                print(f"{rank:<6} {driver_name:<20} {driver_num:<10} {pred_pos:<15.2f} {grid_pos:<10}")
        
        # Save predictions
        output_path = Path(args.output_file)
        if args.show_all:
            all_results.to_csv(output_path, index=False)
            print(f"\nAll {len(all_results)} driver predictions saved to {output_path}")
        else:
            top10.to_csv(output_path, index=False)
            print(f"\nTop 10 predictions saved to {output_path}")
        
        # Show summary statistics
        print(f"\nSummary:")
        print(f"  Best predicted: Position {top10['PredictedPosition'].min():.2f}")
        print(f"  Worst in top 10: Position {top10['PredictedPosition'].max():.2f}")
        print(f"  Average predicted position (top 10): {top10['PredictedPosition'].mean():.2f}")
        
        # If actual positions are available, show ranking accuracy
        if 'ActualPosition' in all_results.columns:
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
            print(f"{'Driver':<20} {'Pred Rank':<12} {'Actual Pos':<12} {'Error':<10} {'Status':<15}")
            print("-" * 70)
            
            for _, row in top10.iterrows():
                if not pd.isna(row.get('ActualPosition')):
                    pred_rank = row['Rank']  # Predicted rank in top 10 (1-10)
                    actual_pos = int(row['ActualPosition'])  # Actual finishing position (1-20)
                    error = abs(pred_rank - actual_pos)
                    driver_name = row.get('DriverName', f"Driver {row['DriverNumber']}")
                    ranking_errors.append((driver_name, pred_rank, actual_pos, error))
                    
                    # Determine status
                    if error == 0:
                        status = "Exact"
                        exact_matches += 1
                    elif error == 1:
                        status = "Close"
                    elif error == 2:
                        status = "Good"
                    elif error <= 3:
                        status = "Fair"
                    else:
                        status = "Poor"
                    
                    if error <= 1:
                        within_1 += 1
                    if error <= 2:
                        within_2 += 1
                    if error <= 3:
                        within_3 += 1
                    
                    print(f"{driver_name:<20} {pred_rank:<12} {actual_pos:<12} {error:<10} {status:<15}")
            
            print("-" * 70)
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
