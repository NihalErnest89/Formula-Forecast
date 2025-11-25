"""
Prediction script for F1 Position Prediction using Classification (Top 10 Only).
Uses Hungarian algorithm to ensure unique position assignments.
Includes interactive race selection like the regression version.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
import pickle
import sys
import argparse
import warnings

# Import from top20/train.py for shared utilities
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import importlib.util

# Import feature calculation and interactive selection from regression predict.py
spec_reg = importlib.util.spec_from_file_location("reg_predict", parent_dir / "top10" / "predict.py")
reg_predict_module = importlib.util.module_from_spec(spec_reg)
spec_reg.loader.exec_module(reg_predict_module)

# Reuse functions from regression predict.py
calculate_future_race_features = reg_predict_module.calculate_future_race_features
select_race_interactive = reg_predict_module.select_race_interactive
get_future_races = reg_predict_module.get_future_races
FEATURE_COLS = reg_predict_module.FEATURE_COLS

# Feature columns - will be determined dynamically based on what's in the data
FEATURE_COLS = ['SeasonPoints', 'SeasonStanding', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                'ConstructorStanding', 'ConstructorTrackAvg', 'GridPosition', 'RecentForm', 'TrackType']


class F1ClassificationNetwork(nn.Module):
    """
    Deep Neural Network for F1 Position Prediction (Classification).
    Must match training architecture.
    """
    
    def __init__(self, input_size=9, hidden_sizes=[256, 128, 64], dropout_rate=0.3):
        super(F1ClassificationNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers - match saved model: Linear -> BatchNorm -> ReLU -> Dropout
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # BatchNorm before activation (matches training)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer: 10 classes for positions 1-10
        layers.append(nn.Linear(prev_size, 10))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)


def load_model(model_path: Path, scaler_path: Path, device, input_size=9):
    """Load trained classification model and scaler. Handles both single and ensemble models."""
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Check if ensemble models exist
    models_dir = model_path.parent
    ensemble_model_0 = models_dir / 'f1_classifier_top10_ensemble_0.pth'
    ensemble_model_1 = models_dir / 'f1_classifier_top10_ensemble_1.pth'
    ensemble_model_2 = models_dir / 'f1_classifier_top10_ensemble_2.pth'
    
    if ensemble_model_0.exists() and ensemble_model_1.exists() and ensemble_model_2.exists():
        # Load ensemble models
        print("Loading ensemble models (3 models)...")
        models = []
        for i, ensemble_path in enumerate([ensemble_model_0, ensemble_model_1, ensemble_model_2]):
            model = F1ClassificationNetwork(input_size=input_size, 
                                          hidden_sizes=[256, 128, 64], 
                                          dropout_rate=0.3)
            model.load_state_dict(torch.load(ensemble_path, map_location=device))
            model.eval()
            model.to(device)
            models.append(model)
            print(f"  Loaded ensemble model {i+1}/3")
        print("Ensemble models loaded successfully")
        return models, scaler
    else:
        # Load single model
        model = F1ClassificationNetwork(input_size=input_size, 
                                        hidden_sizes=[256, 128, 64], 
                                        dropout_rate=0.3)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        return model, scaler


def predict_with_hungarian(model, X_scaled, device):
    """
    Predict positions using Hungarian algorithm to ensure unique assignments.
    Supports ensemble models (list of models).
    
    Args:
        model: Trained classification model or list of models (ensemble)
        X_scaled: Scaled feature matrix (n_drivers x n_features) as numpy array
        device: PyTorch device
    
    Returns:
        Array of assigned positions (1-10) for each driver
    """
    # Handle ensemble models
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
            # Get probabilities for each driver
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
    
    n_drivers = len(X_scaled)
    
    # If we have exactly 10 drivers, use Hungarian algorithm
    if n_drivers == 10:
        # Cost matrix: negative log probabilities (we want to maximize probability)
        cost_matrix = -np.log(probs + 1e-10)  # Add small epsilon to avoid log(0)
        
        # Hungarian algorithm: find optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Assign positions (col_indices are position classes 0-9, convert to 1-10)
        positions = np.zeros(n_drivers, dtype=int)
        # row_indices[i] is the driver index, col_indices[i] is the position class (0-9)
        # linear_sum_assignment returns matched pairs, so we iterate through them
        for i in range(len(row_indices)):
            driver_idx = row_indices[i]
            position_class = col_indices[i]
            positions[driver_idx] = position_class + 1  # Convert 0-9 to 1-10
        
        # Verify all positions are assigned (should be 1-10)
        if not (np.all(positions > 0) and np.all(positions <= 10)):
            print(f"Warning: Invalid positions from Hungarian algorithm: {positions}")
            print(f"  row_indices: {row_indices}, col_indices: {col_indices}")
        
        return positions
    
    # If we have fewer than 10 drivers, assign greedily
    elif n_drivers < 10:
        positions = np.zeros(n_drivers, dtype=int)
        used_positions = set()
        
        # Sort drivers by their max probability
        driver_max_probs = np.max(probs, axis=1)
        driver_order = np.argsort(-driver_max_probs)  # Descending order
        
        for driver_idx in driver_order:
            # Get probabilities for this driver
            driver_probs = probs[driver_idx]
            
            # Find best available position (highest probability among available positions)
            best_pos = None
            best_prob = -1
            for pos_class in range(10):
                pos = pos_class + 1  # Convert to position 1-10
                if pos not in used_positions:
                    if driver_probs[pos_class] > best_prob:
                        best_prob = driver_probs[pos_class]
                        best_pos = pos
            
            if best_pos is not None:
                positions[driver_idx] = best_pos
                used_positions.add(best_pos)
            else:
                # This shouldn't happen, but if it does, assign sequentially
                print(f"Warning: Could not assign position for driver {driver_idx}, assigning sequentially")
                for pos in range(1, 11):
                    if pos not in used_positions:
                        positions[driver_idx] = pos
                        used_positions.add(pos)
                        break
        
        # Verify all positions are assigned
        if np.any(positions == 0):
            print(f"Warning: Some positions are still 0 after greedy assignment: {positions}")
        
        return positions
    
    # If we have more than 10 drivers, assign top 10 based on probabilities
    else:
        # Get top 10 drivers by max probability
        driver_max_probs = np.max(probs, axis=1)
        top10_indices = np.argsort(-driver_max_probs)[:10]
        
        # Use Hungarian algorithm for top 10
        top10_probs = probs[top10_indices]
        cost_matrix = -np.log(top10_probs + 1e-10)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Assign positions
        positions = np.zeros(n_drivers, dtype=int)
        # row_indices[i] is the index within top10_indices (0-9), col_indices[i] is the position class (0-9)
        # We need to map row_indices back to the original driver indices
        for i in range(len(row_indices)):
            # row_indices[i] is the position in the top10_indices array (0-9)
            # top10_indices[row_indices[i]] is the original driver index
            driver_idx = top10_indices[row_indices[i]]
            position_class = col_indices[i]
            positions[driver_idx] = position_class + 1  # Convert 0-9 to 1-10
        
        # Verify top 10 positions are assigned
        top10_positions = positions[top10_indices]
        if not (np.all(top10_positions > 0) and np.all(top10_positions <= 10)):
            print(f"Warning: Invalid positions for top 10 drivers: {top10_positions}")
            print(f"  top10_indices: {top10_indices}")
            print(f"  row_indices: {row_indices}, col_indices: {col_indices}")
            print(f"  positions array: {positions}")
            # Fix any missing positions
            used_positions = set(positions[top10_indices])
            missing_positions = set(range(1, 11)) - used_positions
            if missing_positions:
                print(f"  Missing positions: {missing_positions}")
                # Assign missing positions to drivers with 0
                for driver_idx in top10_indices:
                    if positions[driver_idx] == 0 and missing_positions:
                        positions[driver_idx] = missing_positions.pop()
        
        return positions


def predict_race_top10_classification(drivers_df: pd.DataFrame, model, scaler, device):
    """
    Predict top 10 positions for a race using classification model.
    
    Args:
        drivers_df: DataFrame with driver features (must include FEATURE_COLS)
        model: Trained classification model
        scaler: Fitted StandardScaler
        device: PyTorch device
    
    Returns:
        DataFrame with predictions, sorted by predicted position
        Only returns top 10 predicted drivers (positions 1-10)
    """
    # Select and prepare features - use only features that exist in the dataframe
    available_features = [f for f in FEATURE_COLS if f in drivers_df.columns]
    if len(available_features) == 0:
        raise ValueError("No valid feature columns found in drivers_df!")
    
    X = drivers_df[available_features].copy()
    
    # Handle missing values
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(median_val)
    
    # Convert to numpy array BEFORE scaling to avoid feature name warnings
    X_array = X.values
    
    # Scale features (suppress warnings about feature names)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        X_scaled = scaler.transform(X_array)
    
    # Predict positions using Hungarian algorithm
    predicted_positions = predict_with_hungarian(model, X_scaled, device)
    
    # Create results DataFrame
    results = drivers_df.copy()
    results['PredictedPosition'] = predicted_positions
    
    # Filter out invalid positions (0 or > 10) - these shouldn't happen but check anyway
    valid_mask = (results['PredictedPosition'] > 0) & (results['PredictedPosition'] <= 10)
    if not valid_mask.all():
        invalid_count = (~valid_mask).sum()
        print(f"Warning: {invalid_count} drivers have invalid predicted positions (0 or >10)")
        # For debugging: show what went wrong
        invalid_drivers = results[~valid_mask]
        if len(invalid_drivers) > 0:
            print(f"Invalid positions: {invalid_drivers[['DriverName', 'PredictedPosition']].to_string()}")
    
    # Sort by predicted position and return only top 10
    results = results[valid_mask].sort_values('PredictedPosition').reset_index(drop=True)
    results = results.head(10)
    
    return results


def main():
    """Main function for classification predictions."""
    parser = argparse.ArgumentParser(description='F1 Race Position Prediction (Classification - Top 10)')
    parser.add_argument('--input-file', type=str, help='CSV file with driver features for a race')
    parser.add_argument('--output-file', type=str, default='predictions_classification.csv', help='Output file for predictions')
    parser.add_argument('--model-dir', type=str, default='models/top10_classification', help='Directory containing model files')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode: select year and race from available options')
    
    args = parser.parse_args()
    
    print("F1 Position Prediction (Classification - Top 10)")
    print("=" * 70)
    
    # Set device
    device = torch.device('cpu')
    
    # Load model
    script_dir = Path(__file__).parent
    if args.model_dir:
        models_dir = Path(args.model_dir)
    else:
        models_dir = script_dir.parent / 'models' / 'top10_classification'
    
    model_path = models_dir / 'f1_classifier_top10.pth'
    scaler_path = models_dir / 'scaler_top10_classification.pkl'
    
    if not model_path.exists() or not scaler_path.exists():
        print(f"Error: Model files not found. Please train the model first.")
        print(f"Expected paths:")
        print(f"  Model: {model_path}")
        print(f"  Scaler: {scaler_path}")
        return
    
    print(f"Loading model from {model_path}")
    model, scaler = load_model(model_path, scaler_path, device, input_size=9)
    print("Model loaded successfully!")
    
    # Load data
    data_dir = script_dir.parent / 'data'
    test_data_path = data_dir / 'test_data.csv'
    training_data_path = data_dir / 'training_data.csv'
    
    if not test_data_path.exists():
        print(f"Error: Test data not found at {test_data_path}")
        print("Run collect_data.py first to generate test data")
        return
    
    test_df = pd.read_csv(test_data_path)
    training_df = None
    if training_data_path.exists():
        training_df = pd.read_csv(training_data_path)
        # Filter out Pre-Season Testing
        if 'EventName' in training_df.columns:
            training_df = training_df[~training_df['EventName'].str.contains('Pre-Season|Pre Season|Testing', case=False, na=False)].copy()
    
    # Make predictions
    if args.input_file:
        # Predict from input file
        print(f"\nLoading driver data from {args.input_file}")
        df = pd.read_csv(args.input_file)
        predictions = predict_race_top10_classification(df, model, scaler, device)
        
        # Save results
        predictions.to_csv(args.output_file, index=False)
        print(f"\nPredictions saved to {args.output_file}")
        
        # Display results
        print("\n" + "=" * 70)
        print("PREDICTED TOP 10")
        print("=" * 70)
        display_cols = ['PredictedPosition', 'DriverName', 'TeamName', 'SeasonPoints', 'SeasonStanding']
        available_cols = [col for col in display_cols if col in predictions.columns]
        print(predictions[available_cols].head(10).to_string(index=False))
        
    elif args.interactive or (not args.input_file and not args.interactive):
        # Interactive mode (default)
        print(f"\nLoading test data from {test_data_path}")
        result = select_race_interactive(test_df, training_df)
        
        if result[0] is None:
            print("\nSelection cancelled. Exiting.")
            return
        
        df, input_source, is_future_race = result
        
        # Check if "all" mode was selected
        if is_future_race == 'all':
            # Process all races
            all_race_data = df
            print(f"\n{'='*70}")
            print(f"PREDICTING ALL {len(all_race_data)} RACES")
            print(f"{'='*70}\n")
            
            all_predictions = []
            for race_idx, (race_df, race_input_source, is_future) in enumerate(all_race_data, 1):
                try:
                    predictions = predict_race_top10_classification(race_df, model, scaler, device)
                    
                    # Check if predictions is empty
                    if len(predictions) == 0:
                        print(f"{race_input_source}: No predictions generated (empty DataFrame)")
                        continue
                    
                    predictions['InputSource'] = race_input_source
                    predictions['Rank'] = range(1, len(predictions) + 1)
                    all_predictions.append(predictions)
                    
                    # Display formatted table for each race
                    print(f"\n{'='*130}")
                    print(f"{race_input_source}")
                    print(f"{'='*130}")
                    
                    # Debug: check if we have the required columns
                    if 'DriverName' not in predictions.columns and 'DriverNumber' in predictions.columns:
                        # Try to get driver names from DriverNumber if DriverName is missing
                        pass  # Will use fallback in the loop
                    
                    if 'ActualPosition' in predictions.columns:
                        # Show table with actual positions
                        print(f"{'Rank':<6} {'Driver':<12} {'Pred':<6} {'Actual':<8} {'Error':<7} {'Status':<8} {'SeasPts':<8} {'SeasSt':<8} {'SeasAvg':<8} {'TrackAvg':<9} {'ConstrSt':<9} {'ConstrTrk':<9} {'Form':<7} {'TrackType':<9}")
                        print("-" * 130)
                        
                        def get_status(error: int) -> str:
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
                        
                        for _, row in predictions.iterrows():
                            try:
                                driver_name = row.get('DriverName', f"Driver {row.get('DriverNumber', 'N/A')}")
                                rank = row.get('Rank', row.name + 1)
                                pred_pos = int(row['PredictedPosition'])
                                actual_pos = int(row['ActualPosition']) if not pd.isna(row.get('ActualPosition')) else 'N/A'
                                error = abs(pred_pos - row['ActualPosition']) if not pd.isna(row.get('ActualPosition')) else 'N/A'
                                status = get_status(int(error)) if error != 'N/A' else 'N/A'
                                
                                season_pts = row.get('SeasonPoints', 0)
                                season_st = row.get('SeasonStanding', 20)
                                season_avg = row.get('SeasonAvgFinish', 0)
                                track_avg = row.get('HistoricalTrackAvgPosition', 15.0)
                                if pd.isna(track_avg):
                                    track_avg = 15.0
                                constr_st = row.get('ConstructorStanding', 10)
                                constr_track_avg = row.get('ConstructorTrackAvg', 10.0)
                                if pd.isna(constr_track_avg):
                                    constr_track_avg = 10.0
                                recent_form = row.get('RecentForm', 0)
                                track_type = row.get('TrackType', 0)
                                
                                # Format values
                                season_pts_str = f"{season_pts:.0f}" if not pd.isna(season_pts) else "N/A"
                                season_st_str = f"{season_st:.0f}" if not pd.isna(season_st) else "N/A"
                                season_avg_str = f"{season_avg:.2f}" if not pd.isna(season_avg) else "N/A"
                                track_avg_str = f"{track_avg:.2f}"
                                constr_st_str = f"{constr_st:.0f}" if not pd.isna(constr_st) else "N/A"
                                constr_track_avg_str = f"{constr_track_avg:.2f}"
                                recent_form_str = f"{recent_form:.2f}" if not pd.isna(recent_form) else "N/A"
                                track_type_str = "Street" if track_type == 1 else "Permanent"
                                error_str = f"{error:.0f}" if error != 'N/A' else "N/A"
                                
                                print(f"{rank:<6} {driver_name:<12} {pred_pos:<6} {actual_pos:<8} {error_str:<7} {status:<8} {season_pts_str:<8} {season_st_str:<8} {season_avg_str:<8} {track_avg_str:<9} {constr_st_str:<9} {constr_track_avg_str:<9} {recent_form_str:<7} {track_type_str:<9}")
                            except Exception as row_error:
                                print(f"  Error displaying row: {row_error}")
                                continue
                        
                        print("-" * 130)
                        
                        # Calculate and display accuracy metrics
                        top10_mask = (predictions['ActualPosition'] >= 1) & (predictions['ActualPosition'] <= 10)
                        top10_predictions = predictions[top10_mask].copy()
                        
                        if len(top10_predictions) > 0:
                            errors = np.abs(top10_predictions['PredictedPosition'] - top10_predictions['ActualPosition'])
                            mae = errors.mean()
                            exact = (errors == 0).mean() * 100
                            within_1 = (errors <= 1).mean() * 100
                            within_3 = (errors <= 3).mean() * 100
                            
                            print(f"Accuracy (Top 10 finishers only, {len(top10_predictions)} drivers):")
                            print(f"  MAE: {mae:.2f} positions | Exact: {exact:.1f}% | Within 1: {within_1:.1f}% | Within 3: {within_3:.1f}%")
                        else:
                            print(f"No top 10 finishers in this race to evaluate")
                    else:
                        # Future race - show table without actual positions
                        print(f"{'Rank':<6} {'Driver':<12} {'Pred':<6} {'SeasPts':<8} {'SeasSt':<8} {'SeasAvg':<8} {'TrackAvg':<9} {'ConstrSt':<9} {'ConstrTrk':<9} {'Form':<7} {'TrackType':<9}")
                        print("-" * 130)
                        
                        for _, row in predictions.iterrows():
                            driver_name = row.get('DriverName', f"Driver {row.get('DriverNumber', 'N/A')}")
                            rank = row['Rank']
                            pred_pos = int(row['PredictedPosition'])
                            
                            season_pts = row.get('SeasonPoints', 0)
                            season_st = row.get('SeasonStanding', 20)
                            season_avg = row.get('SeasonAvgFinish', 0)
                            track_avg = row.get('HistoricalTrackAvgPosition', 15.0)
                            if pd.isna(track_avg):
                                track_avg = 15.0
                            constr_st = row.get('ConstructorStanding', 10)
                            constr_track_avg = row.get('ConstructorTrackAvg', 10.0)
                            if pd.isna(constr_track_avg):
                                constr_track_avg = 10.0
                            recent_form = row.get('RecentForm', 0)
                            track_type = row.get('TrackType', 0)
                            
                            # Format values
                            season_pts_str = f"{season_pts:.0f}" if not pd.isna(season_pts) else "N/A"
                            season_st_str = f"{season_st:.0f}" if not pd.isna(season_st) else "N/A"
                            season_avg_str = f"{season_avg:.2f}" if not pd.isna(season_avg) else "N/A"
                            track_avg_str = f"{track_avg:.2f}"
                            constr_st_str = f"{constr_st:.0f}" if not pd.isna(constr_st) else "N/A"
                            constr_track_avg_str = f"{constr_track_avg:.2f}"
                            recent_form_str = f"{recent_form:.2f}" if not pd.isna(recent_form) else "N/A"
                            track_type_str = "Street" if track_type == 1 else "Permanent"
                            
                            print(f"{rank:<6} {driver_name:<12} {pred_pos:<6} {season_pts_str:<8} {season_st_str:<8} {season_avg_str:<8} {track_avg_str:<9} {constr_st_str:<9} {constr_track_avg_str:<9} {recent_form_str:<7} {track_type_str:<9}")
                        
                        print("-" * 130)
                        print("(Future race - no actual results available)")
                    
                    print()  # Blank line between races
                except Exception as e:
                    print(f"  Error predicting {race_input_source}: {e}")
                    import traceback
                    traceback.print_exc()
            
            if all_predictions:
                results_df = pd.concat(all_predictions, ignore_index=True)
                output_path = models_dir / 'classification_predictions_all.csv'
                results_df.to_csv(output_path, index=False)
                print(f"All predictions saved to {output_path}")
        else:
            # Single race
            print(f"\nPredicting for: {input_source}")
            predictions = predict_race_top10_classification(df, model, scaler, device)
            
            # Add Rank column (1-10)
            predictions['Rank'] = range(1, len(predictions) + 1)
            
            # Display results in formatted table
            print("\n" + "=" * 130)
            print("PREDICTED TOP 10 FINISHERS")
            print("=" * 130)
            
            if 'ActualPosition' in predictions.columns:
                # Show table with actual positions for comparison
                print(f"{'Rank':<6} {'Driver':<12} {'Pred':<6} {'Actual':<8} {'Error':<7} {'Status':<8} {'SeasPts':<8} {'SeasSt':<8} {'SeasAvg':<8} {'TrackAvg':<9} {'ConstrSt':<9} {'ConstrTrk':<9} {'Form':<7} {'TrackType':<9}")
                print("-" * 130)
                
                # Get status function from regression predict
                def get_status(error: int) -> str:
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
                
                for _, row in predictions.iterrows():
                    driver_name = row.get('DriverName', f"Driver {row.get('DriverNumber', 'N/A')}")
                    rank = row['Rank']
                    pred_pos = int(row['PredictedPosition'])
                    actual_pos = int(row['ActualPosition']) if not pd.isna(row.get('ActualPosition')) else 'N/A'
                    error = abs(pred_pos - row['ActualPosition']) if not pd.isna(row.get('ActualPosition')) else 'N/A'
                    status = get_status(int(error)) if error != 'N/A' else 'N/A'
                    
                    season_pts = row.get('SeasonPoints', 0)
                    season_st = row.get('SeasonStanding', 20)
                    season_avg = row.get('SeasonAvgFinish', 0)
                    track_avg = row.get('HistoricalTrackAvgPosition', 15.0)
                    if pd.isna(track_avg):
                        track_avg = 15.0
                    constr_st = row.get('ConstructorStanding', 10)
                    constr_track_avg = row.get('ConstructorTrackAvg', 10.0)
                    if pd.isna(constr_track_avg):
                        constr_track_avg = 10.0
                    recent_form = row.get('RecentForm', 0)
                    track_type = row.get('TrackType', 0)
                    
                    # Format values
                    season_pts_str = f"{season_pts:.0f}" if not pd.isna(season_pts) else "N/A"
                    season_st_str = f"{season_st:.0f}" if not pd.isna(season_st) else "N/A"
                    season_avg_str = f"{season_avg:.2f}" if not pd.isna(season_avg) else "N/A"
                    track_avg_str = f"{track_avg:.2f}"
                    constr_st_str = f"{constr_st:.0f}" if not pd.isna(constr_st) else "N/A"
                    constr_track_avg_str = f"{constr_track_avg:.2f}"
                    recent_form_str = f"{recent_form:.2f}" if not pd.isna(recent_form) else "N/A"
                    track_type_str = "Street" if track_type == 1 else "Permanent"
                    error_str = f"{error:.0f}" if error != 'N/A' else "N/A"
                    
                    print(f"{rank:<6} {driver_name:<12} {pred_pos:<6} {actual_pos:<8} {error_str:<7} {status:<8} {season_pts_str:<8} {season_st_str:<8} {season_avg_str:<8} {track_avg_str:<9} {constr_st_str:<9} {constr_track_avg_str:<9} {recent_form_str:<7} {track_type_str:<9}")
                
                print("-" * 130)
                
                # Calculate accuracy if actual positions available
                top10_mask = (predictions['ActualPosition'] >= 1) & (predictions['ActualPosition'] <= 10)
                top10_predictions = predictions[top10_mask].copy()
                
                if len(top10_predictions) > 0:
                    errors = np.abs(top10_predictions['PredictedPosition'] - top10_predictions['ActualPosition'])
                    mae = errors.mean()
                    exact = (errors == 0).mean() * 100
                    within_1 = (errors <= 1).mean() * 100
                    within_3 = (errors <= 3).mean() * 100
                    
                    print(f"\nAccuracy Metrics (Top 10 finishers only, {len(top10_predictions)} drivers):")
                    print(f"  MAE: {mae:.2f} positions")
                    print(f"  Exact Accuracy: {exact:.1f}%")
                    print(f"  Within 1 Position: {within_1:.1f}%")
                    print(f"  Within 3 Positions: {within_3:.1f}%")
                else:
                    print(f"\nNo top 10 finishers in this race to evaluate")
            else:
                # Future race - show table without actual positions
                print(f"{'Rank':<6} {'Driver':<12} {'Pred':<6} {'SeasPts':<8} {'SeasSt':<8} {'SeasAvg':<8} {'TrackAvg':<9} {'ConstrSt':<9} {'ConstrTrk':<9} {'Form':<7} {'TrackType':<9}")
                print("-" * 130)
                
                for _, row in predictions.iterrows():
                    driver_name = row.get('DriverName', f"Driver {row.get('DriverNumber', 'N/A')}")
                    rank = row['Rank']
                    pred_pos = int(row['PredictedPosition'])
                    
                    season_pts = row.get('SeasonPoints', 0)
                    season_st = row.get('SeasonStanding', 20)
                    season_avg = row.get('SeasonAvgFinish', 0)
                    track_avg = row.get('HistoricalTrackAvgPosition', 15.0)
                    if pd.isna(track_avg):
                        track_avg = 15.0
                    constr_st = row.get('ConstructorStanding', 10)
                    constr_track_avg = row.get('ConstructorTrackAvg', 10.0)
                    if pd.isna(constr_track_avg):
                        constr_track_avg = 10.0
                    recent_form = row.get('RecentForm', 0)
                    track_type = row.get('TrackType', 0)
                    
                    # Format values
                    season_pts_str = f"{season_pts:.0f}" if not pd.isna(season_pts) else "N/A"
                    season_st_str = f"{season_st:.0f}" if not pd.isna(season_st) else "N/A"
                    season_avg_str = f"{season_avg:.2f}" if not pd.isna(season_avg) else "N/A"
                    track_avg_str = f"{track_avg:.2f}"
                    constr_st_str = f"{constr_st:.0f}" if not pd.isna(constr_st) else "N/A"
                    constr_track_avg_str = f"{constr_track_avg:.2f}"
                    recent_form_str = f"{recent_form:.2f}" if not pd.isna(recent_form) else "N/A"
                    track_type_str = "Street" if track_type == 1 else "Permanent"
                    
                    print(f"{rank:<6} {driver_name:<12} {pred_pos:<6} {season_pts_str:<8} {season_st_str:<8} {season_avg_str:<8} {track_avg_str:<9} {constr_st_str:<9} {constr_track_avg_str:<9} {recent_form_str:<7} {track_type_str:<9}")
                
                print("-" * 130)
            
            # Save results
            output_path = models_dir / args.output_file
            predictions.to_csv(output_path, index=False)
            print(f"\nPredictions saved to {output_path}")


if __name__ == '__main__':
    main()
