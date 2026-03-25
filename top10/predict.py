"""
Inference script for F1 Predictions (Top 10 Model).
Loads trained top 10 model (trained on positions 1-10 only) and makes predictions for race positions.
Ranks drivers and displays predicted top 10.
Predictions are clipped to range 1-10 to match training data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys

# Import from refactored modules
sys.path.insert(0, str(Path(__file__).parent))

from config import FEATURE_COLS
from evaluation import get_status, calculate_filtered_accuracy
from model_loader import F1NeuralNetwork, load_model, make_predictions, handle_nan_values
from feature_calculation import (
    calculate_future_race_features,
    recalculate_features_from_state,
    update_state_with_actual_results,
    update_state_with_predictions
)
from race_selection import get_future_races, select_race_interactive


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
    # Base features (current feature set, from config.FEATURE_COLS)
    # Check if required columns exist
    missing_cols = [col for col in FEATURE_COLS if col not in drivers_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Prepare features using base features
    X = drivers_df[FEATURE_COLS].values
    X = handle_nan_values(X)
    
    # Verify we have expected number of features
    expected_features = len(FEATURE_COLS)
    if X.shape[1] != expected_features:
        raise ValueError(f"Expected {expected_features} features, but got {X.shape[1]} features")
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Verify model expects correct number of features (fail fast if old model detected)
    if model_type == 'neural_network':
        if isinstance(model, list):
            model_to_check = model[0]
        else:
            model_to_check = model
        model_expected_features = model_to_check.network[0].weight.shape[1]
        if model_expected_features != expected_features:
            raise ValueError(
                f"\n{'='*70}\n"
                f"ERROR: Old model detected! Model expects {model_expected_features} features, "
                f"but code is configured for {expected_features} features.\n"
                f"\nThis is an old model file. Please delete it and retrain:\n"
                f"  1. Delete old model files in models/ directory:\n"
                f"     - f1_predictor_model_top10.pth\n"
                f"     - f1_predictor_model_top10_ensemble_*.pth\n"
                f"     - scaler_top10.pkl\n"
                f"  2. Retrain with: python top10/train.py\n"
                f"{'='*70}"
            )
    
    predicted_positions = make_predictions(X_scaled, model, model_type, device)
    
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
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        if isinstance(e, FileNotFoundError):
            print(f"\nMake sure you've trained the model first:")
            if args.model_type == 'neural_network':
                print("  python top10/train.py")
            else:
                print("  python top10/train_rf.py")
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
            print("  - AvgGridPosition (season-specific average grid position)")
            print("  - RecentForm")
            print("  - TrackType")
            print("  - DriverNumber (optional)")
            print("  - DriverName (optional)")
            print("\nExample:")
            print("  python predict.py --input-file race_drivers.csv")
            print("\nOr run collect_data.py first to generate test data.")
            return
        
        test_df = pd.read_csv(test_data_path)
        
        # Filter out Pre-Season Testing (not a real race)
        if 'EventName' in test_df.columns:
            test_df = test_df[~test_df['EventName'].str.contains('Pre-Season|Pre Season|Testing', case=False, na=False)].copy()
        
        # Load training data for historical track averages (if available)
        training_data_path = Path('data') / 'training_data.csv'
        training_df = None
        if training_data_path.exists():
            try:
                training_df = pd.read_csv(training_data_path)
                # Filter out Pre-Season Testing (not a real race)
                if 'EventName' in training_df.columns:
                    training_df = training_df[~training_df['EventName'].str.contains('Pre-Season|Pre Season|Testing', case=False, na=False)].copy()
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
                # All race DataFrames are ready (TrackID removed, no processing needed)
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
                        # Filter out Pre-Season Testing (not a real race)
                        if 'EventName' in training_df_all.columns:
                            training_df_all = training_df_all[~training_df_all['EventName'].str.contains('Pre-Season|Pre Season|Testing', case=False, na=False)].copy()
                    except Exception:
                        pass
                
                # Ensure test_df is available for actual race results (reload if needed)
                test_data_path = script_dir.parent / 'data' / 'test_data.csv'
                test_df_for_races = None
                if test_data_path.exists():
                    try:
                        test_df_for_races = pd.read_csv(test_data_path)
                        # Filter out Pre-Season Testing (not a real race)
                        if 'EventName' in test_df_for_races.columns:
                            test_df_for_races = test_df_for_races[~test_df_for_races['EventName'].str.contains('Pre-Season|Pre Season|Testing', case=False, na=False)].copy()
                    except Exception:
                        pass
                
                for race_idx, (race_df, race_input_source, is_future) in enumerate(all_race_data, 1):
                    print(f"\n{'='*70}")
                    print(f"Race {race_idx}/{len(all_race_data)}: {race_input_source}")
                    print(f"{'='*70}")
                    print(f"  Found {len(race_df)} drivers")
                    
                    # For the first race, initialize state with 0 season points
                    if current_state_df is None and not is_future:
                        # First actual race - initialize state with 0 points for this season
                        # Reset season points to 0 (test_df has cumulative points, we need to start fresh)
                        current_state_df = race_df.copy()
                        # Reset cumulative features to 0/NaN for new season
                        current_state_df['SeasonPoints'] = 0
                        current_state_df['SeasonStanding'] = 20  # Default to worst position for new season
                        current_state_df['SeasonAvgFinish'] = np.nan
                        current_state_df['RecentForm'] = np.nan
                        # Also reset ConstructorPoints to 0 for new season
                        if 'ConstructorPoints' in current_state_df.columns:
                            current_state_df['ConstructorPoints'] = 0
                    
                    # For actual races, use updated state if available (from previous races)
                    # For future races, progressively update features based on previous race
                    if current_state_df is not None:
                        # Recalculate features from updated state (works for both actual and future races)
                        track_name = race_df['EventName'].iloc[0] if 'EventName' in race_df.columns else race_input_source.split('(')[0].strip()
                        # Get year and round from race_df for constructor standings calculation
                        race_year = race_df['Year'].iloc[0] if 'Year' in race_df.columns else None
                        race_round = race_df['RoundNumber'].iloc[0] if 'RoundNumber' in race_df.columns else None
                        race_df = recalculate_features_from_state(
                            race_df, current_state_df, track_name, training_df_all,
                            test_df_for_races, race_year, race_round
                        )
                    elif is_future:
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
                        
                        # Update current state BEFORE displaying (so we show points AFTER the race)
                        # For actual races: use actual results to update state
                        # For future races: use predictions as simulated results
                        if current_state_df is None:
                            current_state_df = race_df.copy()
                        
                        if is_future:
                            # Future race: use predictions to simulate results
                            current_state_df = update_state_with_predictions(current_state_df, all_results)
                        else:
                            # Actual race: use actual results to update state
                            # Get actual positions from test_df for this specific race (more reliable than all_results)
                            race_year = race_df['Year'].iloc[0] if 'Year' in race_df.columns else None
                            race_round = race_df['RoundNumber'].iloc[0] if 'RoundNumber' in race_df.columns else None
                            race_event = race_df['EventName'].iloc[0] if 'EventName' in race_df.columns else None
                            
                            # Use test_df_for_races loaded above, or try to get from main scope
                            test_df_for_update = test_df_for_races
                            
                            if race_year and race_round and race_event and test_df_for_update is not None:
                                # Get actual race results from test_df for this specific race
                                actual_race_results = test_df_for_update[
                                    (test_df_for_update['Year'] == race_year) &
                                    (test_df_for_update['RoundNumber'] == race_round) &
                                    (test_df_for_update['EventName'] == race_event) &
                                    (test_df_for_update['ActualPosition'].notna())
                                ].copy()
                                
                                if not actual_race_results.empty:
                                    # Use actual positions from test_df
                                    current_state_df = update_state_with_actual_results(
                                        current_state_df, actual_race_results, test_df_for_update, race_year, race_round
                                    )
                                elif 'ActualPosition' in all_results.columns and not all_results['ActualPosition'].isna().all():
                                    # Fallback: use all_results if test_df doesn't have this race
                                    current_state_df = update_state_with_actual_results(
                                        current_state_df, all_results, test_df_for_update, race_year, race_round
                                    )
                            elif 'ActualPosition' in all_results.columns and not all_results['ActualPosition'].isna().all():
                                # Fallback: use all_results if we can't get race info
                                current_state_df = update_state_with_actual_results(
                                    current_state_df, all_results, test_df_for_update, race_year, race_round
                                )
                        
                        # Update all_results with the updated state features (so display shows points AFTER race)
                        if current_state_df is not None:
                            for idx, row in all_results.iterrows():
                                driver_num = row.get('DriverNumber')
                                if driver_num is not None:
                                    state_row = current_state_df[current_state_df['DriverNumber'] == driver_num]
                                    if not state_row.empty:
                                        state_row = state_row.iloc[0]
                                        # Update features in all_results to reflect state AFTER the race
                                        all_results.at[idx, 'SeasonPoints'] = state_row.get('SeasonPoints', row.get('SeasonPoints', 0))
                                        all_results.at[idx, 'SeasonAvgFinish'] = state_row.get('SeasonAvgFinish', row.get('SeasonAvgFinish', np.nan))
                                        all_results.at[idx, 'RecentForm'] = state_row.get('RecentForm', row.get('RecentForm', np.nan))
                            
                            # Update top10 with the updated features
                            for idx, row in top10.iterrows():
                                driver_num = row.get('DriverNumber')
                                if driver_num is not None:
                                    state_row = current_state_df[current_state_df['DriverNumber'] == driver_num]
                                    if not state_row.empty:
                                        state_row = state_row.iloc[0]
                                        top10.at[idx, 'SeasonPoints'] = state_row.get('SeasonPoints', row.get('SeasonPoints', 0))
                                        top10.at[idx, 'SeasonAvgFinish'] = state_row.get('SeasonAvgFinish', row.get('SeasonAvgFinish', np.nan))
                                        top10.at[idx, 'RecentForm'] = state_row.get('RecentForm', row.get('RecentForm', np.nan))
                        
                        # Display ranking accuracy analysis table (for "all" mode)
                        if not race_is_future_check and 'ActualPosition' in all_results.columns:
                            print(f"\n{'='*70}")
                            print(f"RANKING ACCURACY ANALYSIS")
                            print(f"{'='*70}")
                            print(f"Comparing: Predicted Score vs Actual Finishing Position (1-20)")
                            print()
                            print(f"{'Driver':<12} {'PredScore':<9} {'PredPos':<8} {'Actual':<8} {'Error':<7} {'Status':<8} {'AvgGridPos':<10} {'SeasPts':<8} {'SeasAvg':<8} {'TrackAvg':<9} {'ConstrSt':<9} {'ConstrTrk':<9} {'Form':<7} {'Wins':<5} {'TrackType':<9}")
                            print("-" * 145)
                            
                            # Collect data for accuracy calculation
                            predicted_list = []
                            actual_list = []
                            grid_list = []
                            
                            for _, row in top10.iterrows():
                                if not pd.isna(row.get('ActualPosition')):
                                    pred_rank = row['Rank']
                                    pred_score = row.get('PredictedPosition', pred_rank)
                                    actual_pos = int(row['ActualPosition'])
                                    grid_pos = row.get('GridPosition', 'N/A')
                                    if pd.isna(grid_pos):
                                        grid_pos = 'N/A'
                                        grid_pos_float = 10.5
                                    else:
                                        grid_pos_float = float(grid_pos)
                                        grid_pos = int(grid_pos)
                                    error = abs(pred_rank - actual_pos)
                                    driver_name = row.get('DriverName', f"Driver {row['DriverNumber']}")
                                    status = get_status(error)
                                    
                                    # Collect for accuracy calculation (use raw predicted scores, not ranks)
                                    predicted_list.append(pred_score)
                                    actual_list.append(actual_pos)
                                    grid_list.append(grid_pos_float)
                                    
                                    # Get only features used by model (10 features)
                                    season_pts = row.get('SeasonPoints', 0)
                                    season_avg = row.get('SeasonAvgFinish', 0)
                                    track_avg = row.get('HistoricalTrackAvgPosition', 10.0)
                                    if pd.isna(track_avg):
                                        track_avg = 10.0
                                    constr_st = row.get('ConstructorStanding', 0)
                                    constr_track_avg = row.get('ConstructorTrackAvg', 10.0)
                                    if pd.isna(constr_track_avg):
                                        constr_track_avg = 10.0
                                    recent_form = row.get('RecentForm', 0)
                                    career_wins = row.get('CareerWins', 0)
                                    track_type = row.get('TrackType', 0)
                                    # Format values
                                    season_pts_str = f"{season_pts:.0f}" if not pd.isna(season_pts) else "N/A"
                                    season_avg_str = f"{season_avg:.2f}" if not pd.isna(season_avg) else "N/A"
                                    track_avg_str = f"{track_avg:.2f}"  # Always show a number (default 10.0 for rookies)
                                    constr_st_str = f"{constr_st:.0f}" if not pd.isna(constr_st) else "N/A"
                                    constr_track_avg_str = f"{constr_track_avg:.2f}"  # Constructor track average
                                    recent_form_str = f"{recent_form:.2f}" if not pd.isna(recent_form) else "N/A"
                                    track_type_str = "Street" if track_type == 1 else "Permanent"
                                    
                                    print(f"{driver_name:<12} {pred_score:<9.2f} {pred_rank:<8} {actual_pos:<8} {error:<7} {status:<8} {grid_pos:<8} {season_pts_str:<8} {season_avg_str:<8} {track_avg_str:<9} {constr_st_str:<9} {constr_track_avg_str:<9} {recent_form_str:<7} {career_wins:<5.0f} {track_type_str:<9}")
                            
                            # Display filtered top 10 (excluding outliers, re-ranked) before accuracy metrics
                            # Filter out outliers and re-rank
                            filtered_rows = []
                            for _, row in top10.iterrows():
                                if not pd.isna(row.get('ActualPosition')):
                                    actual_pos = int(row['ActualPosition'])
                                    grid_pos = row.get('GridPosition', np.nan)
                                    if pd.isna(grid_pos):
                                        grid_pos_float = 10.5
                                    else:
                                        grid_pos_float = float(grid_pos)
                                    
                                    # Check if outlier (actual > grid + 6)
                                    position_drop = actual_pos - grid_pos_float
                                    if position_drop <= 6:
                                        filtered_rows.append((row, row.get('PredictedPosition', row['Rank'])))
                            
                            if filtered_rows:
                                # Sort by predicted score and assign new ranks
                                filtered_rows.sort(key=lambda x: x[1])
                                
                                print(f"\n{'='*70}")
                                print(f"FILTERED TOP 10 (excluding outliers)")
                                print(f"{'='*70}")
                                print(f"{'Driver':<12} {'PredScore':<9} {'PredPos':<8} {'Actual':<8} {'Error':<7} {'Status':<8} {'AvgGridPos':<10} {'SeasPts':<8} {'SeasAvg':<8} {'TrackAvg':<9} {'ConstrSt':<9} {'ConstrTrk':<9} {'Form':<7} {'Wins':<5} {'TrackType':<9}")
                                print("-" * 145)
                                
                                for new_rank, (row, pred_score) in enumerate(filtered_rows, 1):
                                    actual_pos = int(row['ActualPosition'])
                                    grid_pos = row.get('GridPosition', 'N/A')
                                    if pd.isna(grid_pos):
                                        grid_pos = 'N/A'
                                    else:
                                        grid_pos = int(grid_pos)
                                    error = abs(new_rank - actual_pos)
                                    driver_name = row.get('DriverName', f"Driver {row['DriverNumber']}")
                                    status = get_status(error)
                                    
                                    season_pts = row.get('SeasonPoints', 0)
                                    season_avg = row.get('SeasonAvgFinish', 0)
                                    track_avg = row.get('HistoricalTrackAvgPosition', 10.0)
                                    if pd.isna(track_avg):
                                        track_avg = 10.0
                                    constr_st = row.get('ConstructorStanding', 0)
                                    constr_track_avg = row.get('ConstructorTrackAvg', 10.0)
                                    if pd.isna(constr_track_avg):
                                        constr_track_avg = 10.0
                                    recent_form = row.get('RecentForm', 0)
                                    career_wins = row.get('CareerWins', 0)
                                    track_type = row.get('TrackType', 0)
                                    
                                    season_pts_str = f"{season_pts:.0f}" if not pd.isna(season_pts) else "N/A"
                                    season_avg_str = f"{season_avg:.2f}" if not pd.isna(season_avg) else "N/A"
                                    track_avg_str = f"{track_avg:.2f}"
                                    constr_st_str = f"{constr_st:.0f}" if not pd.isna(constr_st) else "N/A"
                                    constr_track_avg_str = f"{constr_track_avg:.2f}"
                                    recent_form_str = f"{recent_form:.2f}" if not pd.isna(recent_form) else "N/A"
                                    track_type_str = "Street" if track_type == 1 else "Permanent"
                                    
                                    print(f"{driver_name:<12} {pred_score:<9.2f} {new_rank:<8} {actual_pos:<8} {error:<7} {status:<8} {grid_pos:<8} {season_pts_str:<8} {season_avg_str:<8} {track_avg_str:<9} {constr_st_str:<9} {constr_track_avg_str:<9} {recent_form_str:<7} {career_wins:<5.0f} {track_type_str:<9}")
                            
                            # Calculate and display filtered accuracy
                            if predicted_list:
                                accuracy_metrics = calculate_filtered_accuracy(
                                    predicted_list, actual_list, grid_list, outlier_threshold=6
                                )
                                
                                print()
                                print("Ranking Accuracy:")
                                print(f"  Full (all predictions):")
                                print(f"    Exact matches: {accuracy_metrics['full']['exact']:.1f}%")
                                print(f"    Within 1 position: {accuracy_metrics['full']['within_1']:.1f}%")
                                print(f"    Within 2 positions: {accuracy_metrics['full']['within_2']:.1f}%")
                                print(f"    Within 3 positions: {accuracy_metrics['full']['within_3']:.1f}%")
                                print(f"    MAE: {accuracy_metrics['full']['mae']:.2f} positions")
                                
                                if accuracy_metrics['filtered']['outliers_removed'] > 0:
                                    print(f"  Filtered (excluding {accuracy_metrics['filtered']['outliers_removed']} large drops > grid + 6):")
                                    print(f"    Exact matches: {accuracy_metrics['filtered']['exact']:.1f}%")
                                    print(f"    Within 1 position: {accuracy_metrics['filtered']['within_1']:.1f}%")
                                    print(f"    Within 2 positions: {accuracy_metrics['filtered']['within_2']:.1f}%")
                                    print(f"    Within 3 positions: {accuracy_metrics['filtered']['within_3']:.1f}%")
                                    print(f"    MAE: {accuracy_metrics['filtered']['mae']:.2f} positions")
                                    print(f"    Improvement: MAE {accuracy_metrics['full']['mae'] - accuracy_metrics['filtered']['mae']:.2f} positions better, "
                                          f"Within 3: +{accuracy_metrics['filtered']['within_3'] - accuracy_metrics['full']['within_3']:.1f}%")
                            
                            print("-" * 120)
                        else:
                            # Future race - show table with all features
                            print(f"\nPREDICTED TOP 10 FINISHERS")
                            print("-" * 140)
                            print(f"{'Rank':<6} {'Driver':<12} {'Pred':<6} {'SeasPts':<8} {'SeasAvg':<8} "
                                  f"{'TrackAvg':<9} {'ConstrSt':<9} {'ConstrTrk':<9} "
                                  f"{'AvgGridPos':<10} {'Form':<7} {'CareerWins':<11} {'TrackType':<9}")
                            print("-" * 140)
                            
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
                                constr_track_avg = row.get('ConstructorTrackAvg', 10.0)
                                if pd.isna(constr_track_avg):
                                    constr_track_avg = 10.0
                                grid_pos = row.get('GridPosition', 0)
                                recent_form = row.get('RecentForm', 0)
                                career_wins = row.get('CareerWins', 0)
                                track_type = row.get('TrackType', 0)
                                # Format values
                                season_pts_str = f"{season_pts:.0f}" if not pd.isna(season_pts) else "N/A"
                                season_avg_str = f"{season_avg:.2f}" if not pd.isna(season_avg) else "N/A"
                                track_avg_str = f"{track_avg:.2f}"  # Always show a number (default 10.0 for rookies)
                                constr_st_str = f"{constr_st:.0f}" if not pd.isna(constr_st) else "N/A"
                                constr_track_avg_str = f"{constr_track_avg:.2f}"  # Constructor track average
                                grid_pos_str = f"{grid_pos:.0f}" if not pd.isna(grid_pos) else "N/A"
                                recent_form_str = f"{recent_form:.2f}" if not pd.isna(recent_form) else "N/A"
                                career_wins_str = f"{career_wins:.0f}" if not pd.isna(career_wins) else "N/A"
                                track_type_str = "Street" if track_type == 1 else "Permanent"
                                
                                print(f"{rank:<6} {driver_name:<12} {pred_pos:<6.3f} {season_pts_str:<8} "
                                      f"{season_avg_str:<8} {track_avg_str:<9} {constr_st_str:<9} "
                                      f"{constr_track_avg_str:<9} {grid_pos_str:<8} {recent_form_str:<7} "
                                      f"{career_wins_str:<11} {track_type_str:<9}")
                        
                        # Store predictions
                        top10_copy = top10.copy()
                        top10_copy['Race'] = race_input_source
                        all_predictions.append(top10_copy)
                        
                        # Note: State was already updated above before displaying results
                        # Filtered accuracy already displayed above, no need for duplicate Summary/Ranking Accuracy
                        
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
        
        # Display top 10 table only for future races (when actual positions aren't available)
        # For races with actual positions, skip this table since detailed tables below provide all info
        if is_future_race:
            print("\n" + "=" * 70)
            print(f"PREDICTED TOP 10 FINISHERS ({input_source})")
            print("=" * 70)
            # Future race - show features used by model (10 features)
            print(f"{'Rank':<6} {'Driver':<12} {'Pred':<6} {'SeasPts':<8} {'SeasAvg':<8} {'TrackAvg':<9} {'ConstrSt':<9} {'ConstrTrk':<9} {'AvgGridPos':<10} {'Form':<7} {'Wins':<5} {'TrackType':<9}")
            print("-" * 135)

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
                constr_track_avg = row.get('ConstructorTrackAvg', 10.0)
                if pd.isna(constr_track_avg):
                    constr_track_avg = 10.0
                grid_pos_val = row.get('GridPosition', 0)
                recent_form = row.get('RecentForm', 0)
                career_wins = row.get('CareerWins', 0)
                track_type = row.get('TrackType', 0)
                # Format values
                season_pts_str = f"{season_pts:.0f}" if not pd.isna(season_pts) else "N/A"
                season_avg_str = f"{season_avg:.2f}" if not pd.isna(season_avg) else "N/A"
                track_avg_str = f"{track_avg:.2f}"  # Always show a number (default 10.0 for rookies)
                constr_st_str = f"{constr_st:.0f}" if not pd.isna(constr_st) else "N/A"
                constr_track_avg_str = f"{constr_track_avg:.2f}"  # Constructor track average
                grid_pos_str = f"{grid_pos_val:.0f}" if not pd.isna(grid_pos_val) else "N/A"
                recent_form_str = f"{recent_form:.2f}" if not pd.isna(recent_form) else "N/A"
                track_type_str = "Street" if track_type == 1 else "Permanent"
                
                print(f"{rank:<6} {driver_name:<12} {pred_pos:<6.3f} {season_pts_str:<8} {season_avg_str:<8} {track_avg_str:<9} {constr_st_str:<9} {constr_track_avg_str:<9} {grid_pos_str:<8} {recent_form_str:<7} {career_wins:<5.0f} {track_type_str:<9}")
        
        # Save predictions
        output_path = Path(args.output_file)
        if args.show_all:
            all_results.to_csv(output_path, index=False)
            print(f"\nAll {len(all_results)} driver predictions saved to {output_path}")
        else:
            top10.to_csv(output_path, index=False)
            print(f"\nTop 5 predictions saved to {output_path}")
        
        # Show summary statistics only for future races (detailed tables below provide this for races with actual positions)
        if is_future_race:
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
            print(f"Comparing: Predicted Score vs Actual Finishing Position (1-20)")
            print()
            
            # Show detailed comparison table
            print(f"{'Driver':<12} {'PredScore':<9} {'PredPos':<8} {'Actual':<8} {'Error':<7} {'Status':<8} {'AvgGridPos':<10} {'SeasPts':<8} {'SeasAvg':<8} {'TrackAvg':<9} {'ConstrSt':<9} {'ConstrTrk':<9} {'Form':<7} {'Wins':<5} {'TrackType':<9}")
            print("-" * 145)
            
            # Collect data for filtered accuracy calculation
            predicted_list = []
            actual_list = []
            grid_list = []
            
            for _, row in top10.iterrows():
                if not pd.isna(row.get('ActualPosition')):
                    pred_rank = row['Rank']
                    pred_score = row.get('PredictedPosition', pred_rank)
                    actual_pos = int(row['ActualPosition'])
                    grid_pos = row.get('GridPosition', 'N/A')
                    if pd.isna(grid_pos):
                        grid_pos = 'N/A'
                        grid_pos_float = 10.5
                    else:
                        grid_pos_float = float(grid_pos)
                        grid_pos = int(grid_pos)
                    error = abs(pred_rank - actual_pos)
                    driver_name = row.get('DriverName', f"Driver {row['DriverNumber']}")
                    ranking_errors.append((driver_name, pred_rank, actual_pos, error))
                    
                    # Collect for filtered accuracy (use raw predicted scores, not ranks)
                    predicted_list.append(pred_score)
                    actual_list.append(actual_pos)
                    grid_list.append(grid_pos_float)
                    
                    status = get_status(error)
                    if error == 0:
                        exact_matches += 1
                    if error <= 1:
                        within_1 += 1
                    if error <= 2:
                        within_2 += 1
                    if error <= 3:
                        within_3 += 1
                    
                    # Get only features used by model (10 features)
                    season_pts = row.get('SeasonPoints', 0)
                    season_avg = row.get('SeasonAvgFinish', 0)
                    track_avg = row.get('HistoricalTrackAvgPosition', 10.0)
                    if pd.isna(track_avg):
                        track_avg = 10.0
                    constr_st = row.get('ConstructorStanding', 0)
                    constr_track_avg = row.get('ConstructorTrackAvg', 10.0)
                    if pd.isna(constr_track_avg):
                        constr_track_avg = 10.0
                    recent_form = row.get('RecentForm', 0)
                    career_wins = row.get('CareerWins', 0)
                    track_type = row.get('TrackType', 0)
                    # Format values
                    season_pts_str = f"{season_pts:.0f}" if not pd.isna(season_pts) else "N/A"
                    season_avg_str = f"{season_avg:.2f}" if not pd.isna(season_avg) else "N/A"
                    track_avg_str = f"{track_avg:.2f}"  # Always show a number (default 10.0 for rookies)
                    constr_st_str = f"{constr_st:.0f}" if not pd.isna(constr_st) else "N/A"
                    constr_track_avg_str = f"{constr_track_avg:.2f}"  # Constructor track average
                    recent_form_str = f"{recent_form:.2f}" if not pd.isna(recent_form) else "N/A"
                    track_type_str = "Street" if track_type == 1 else "Permanent"
                    
                    print(f"{driver_name:<12} {pred_score:<9.2f} {pred_rank:<8} {actual_pos:<8} {error:<7} {status:<8} {grid_pos:<8} {season_pts_str:<8} {season_avg_str:<8} {track_avg_str:<9} {constr_st_str:<9} {constr_track_avg_str:<9} {recent_form_str:<7} {career_wins:<5.0f} {track_type_str:<9}")
            
            print("-" * 120)
            
            # Display filtered top 10 (excluding outliers, re-ranked) before accuracy metrics
            # Filter out outliers and re-rank
            filtered_rows = []
            for _, row in top10.iterrows():
                if not pd.isna(row.get('ActualPosition')):
                    actual_pos = int(row['ActualPosition'])
                    grid_pos = row.get('GridPosition', np.nan)
                    if pd.isna(grid_pos):
                        grid_pos_float = 10.5
                    else:
                        grid_pos_float = float(grid_pos)
                    
                    # Check if outlier (actual > grid + 6)
                    position_drop = actual_pos - grid_pos_float
                    if position_drop <= 6:
                        filtered_rows.append((row, row.get('PredictedPosition', row['Rank'])))
            
            if filtered_rows:
                # Sort by predicted score and assign new ranks
                filtered_rows.sort(key=lambda x: x[1])
                
                print(f"\n{'='*70}")
                print(f"FILTERED TOP 10 (excluding outliers)")
                print(f"{'='*70}")
                print(f"{'Driver':<12} {'PredScore':<9} {'PredPos':<8} {'Actual':<8} {'Error':<7} {'Status':<8} {'AvgGridPos':<10} {'SeasPts':<8} {'SeasAvg':<8} {'TrackAvg':<9} {'ConstrSt':<9} {'ConstrTrk':<9} {'Form':<7} {'Wins':<5} {'TrackType':<9}")
                print("-" * 145)
                
                for new_rank, (row, pred_score) in enumerate(filtered_rows, 1):
                    actual_pos = int(row['ActualPosition'])
                    grid_pos = row.get('GridPosition', 'N/A')
                    if pd.isna(grid_pos):
                        grid_pos = 'N/A'
                    else:
                        grid_pos = int(grid_pos)
                    error = abs(new_rank - actual_pos)
                    driver_name = row.get('DriverName', f"Driver {row['DriverNumber']}")
                    status = get_status(error)
                    
                    season_pts = row.get('SeasonPoints', 0)
                    season_avg = row.get('SeasonAvgFinish', 0)
                    track_avg = row.get('HistoricalTrackAvgPosition', 10.0)
                    if pd.isna(track_avg):
                        track_avg = 10.0
                    constr_st = row.get('ConstructorStanding', 0)
                    constr_track_avg = row.get('ConstructorTrackAvg', 10.0)
                    if pd.isna(constr_track_avg):
                        constr_track_avg = 10.0
                    recent_form = row.get('RecentForm', 0)
                    career_wins = row.get('CareerWins', 0)
                    track_type = row.get('TrackType', 0)
                    
                    season_pts_str = f"{season_pts:.0f}" if not pd.isna(season_pts) else "N/A"
                    season_avg_str = f"{season_avg:.2f}" if not pd.isna(season_avg) else "N/A"
                    track_avg_str = f"{track_avg:.2f}"
                    constr_st_str = f"{constr_st:.0f}" if not pd.isna(constr_st) else "N/A"
                    constr_track_avg_str = f"{constr_track_avg:.2f}"
                    recent_form_str = f"{recent_form:.2f}" if not pd.isna(recent_form) else "N/A"
                    track_type_str = "Street" if track_type == 1 else "Permanent"
                    
                    print(f"{driver_name:<12} {pred_score:<9.2f} {new_rank:<8} {actual_pos:<8} {error:<7} {status:<8} {grid_pos:<8} {season_pts_str:<8} {season_avg_str:<8} {track_avg_str:<9} {constr_st_str:<9} {constr_track_avg_str:<9} {recent_form_str:<7} {career_wins:<5.0f} {track_type_str:<9}")
            
            # Calculate filtered accuracy
            if predicted_list:
                accuracy_metrics = calculate_filtered_accuracy(
                    predicted_list, actual_list, grid_list, outlier_threshold=6
                )
                
                print(f"\nFiltered Accuracy (excluding large drops > grid + 6):")
                if accuracy_metrics['filtered']['outliers_removed'] > 0:
                    print(f"  Removed {accuracy_metrics['filtered']['outliers_removed']} outliers")
                    print(f"  Exact matches: {accuracy_metrics['filtered']['exact']:.1f}%")
                    print(f"  Within 1 position: {accuracy_metrics['filtered']['within_1']:.1f}%")
                    print(f"  Within 2 positions: {accuracy_metrics['filtered']['within_2']:.1f}%")
                    print(f"  Within 3 positions: {accuracy_metrics['filtered']['within_3']:.1f}%")
                    print(f"  MAE: {accuracy_metrics['filtered']['mae']:.2f} positions")
                    print(f"  Improvement: MAE {accuracy_metrics['full']['mae'] - accuracy_metrics['filtered']['mae']:.2f} positions better, "
                          f"Within 3: +{accuracy_metrics['filtered']['within_3'] - accuracy_metrics['full']['within_3']:.1f}%")
                else:
                    print(f"  No outliers found (all predictions within normal range)")
            
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
