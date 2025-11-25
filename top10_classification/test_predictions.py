"""
Test script to automatically test classification predictions on a few races.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import sys

# Import predict functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from top10_classification.predict import load_model, predict_race_top10_classification
from top10.predict import get_future_races, calculate_future_race_features

def test_predictions(num_races=5):
    """Test predictions on the first few completed races."""
    print("Testing Classification Model Predictions")
    print("=" * 70)
    
    # Set device
    device = torch.device('cpu')
    
    # Load model
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / 'models' / 'top10_classification'
    model_path = models_dir / 'f1_classifier_top10.pth'
    scaler_path = models_dir / 'scaler_top10_classification.pkl'
    
    if not model_path.exists() or not scaler_path.exists():
        print(f"Error: Model files not found at {models_dir}")
        return
    
    print(f"Loading model from {model_path}")
    model, scaler = load_model(model_path, scaler_path, device, input_size=9)
    print("Model loaded successfully!\n")
    
    # Load data
    data_dir = script_dir.parent / 'data'
    test_data_path = data_dir / 'test_data.csv'
    training_data_path = data_dir / 'training_data.csv'
    
    if not test_data_path.exists() or not training_data_path.exists():
        print(f"Error: Data files not found at {data_dir}")
        return
    
    test_df = pd.read_csv(test_data_path)
    training_df = pd.read_csv(training_data_path)
    
    # Get 2025 races
    test_2025 = test_df[test_df['Year'] == 2025].copy()
    
    if len(test_2025) == 0:
        print("No 2025 races found in test data")
        return
    
    # Get unique races
    unique_races = test_2025.groupby(['Year', 'EventName', 'RoundNumber']).first().reset_index()
    unique_races = unique_races.sort_values(['Year', 'RoundNumber'])
    
    # Test first few completed races (where ActualPosition is not NaN)
    completed_races = []
    for _, race_info in unique_races.iterrows():
        race_data = test_2025[
            (test_2025['Year'] == race_info['Year']) &
            (test_2025['EventName'] == race_info['EventName']) &
            (test_2025['RoundNumber'] == race_info['RoundNumber'])
        ]
        if race_data['ActualPosition'].notna().any():
            completed_races.append((race_info, race_data))
        if len(completed_races) >= num_races:
            break
    
    if len(completed_races) == 0:
        print("No completed races found to test")
        return
    
    print(f"Testing {len(completed_races)} races:\n")
    
    all_metrics = []
    
    for race_info, race_data in completed_races:
        year = race_info['Year']
        event_name = race_info['EventName']
        round_num = race_info['RoundNumber']
        
        print(f"{'='*70}")
        print(f"{event_name} ({year}, Round {round_num})")
        print(f"{'='*70}")
        
        try:
            # Calculate features for this race - need to pass test_df, not filtered race_data
            race_features = calculate_future_race_features(
                test_2025, training_df, year, event_name, round_num
            )
            
            if race_features is None or len(race_features) == 0:
                print("  Could not calculate features for this race\n")
                continue
            
            # Make predictions
            predictions = predict_race_top10_classification(
                race_features, model, scaler, device
            )
            
            if len(predictions) == 0:
                print("  No predictions generated\n")
                continue
            
            # Merge with actual results
            predictions = predictions.merge(
                race_data[['DriverName', 'ActualPosition']],
                on='DriverName',
                how='left'
            )
            
            # Filter to only drivers who actually finished in top 10
            top10_finishers = predictions[predictions['ActualPosition'].notna() & 
                                          (predictions['ActualPosition'] <= 10)]
            
            if len(top10_finishers) == 0:
                print("  No top 10 finishers found in predictions\n")
                continue
            
            # Calculate metrics
            errors = np.abs(top10_finishers['PredictedPosition'] - top10_finishers['ActualPosition'])
            mae = errors.mean()
            exact = (errors == 0).sum() / len(top10_finishers)
            within1 = (errors <= 1).sum() / len(top10_finishers)
            within3 = (errors <= 3).sum() / len(top10_finishers)
            
            # Top-3 accuracy: how many of actual top 3 were predicted in top 3
            actual_top3 = top10_finishers.nsmallest(3, 'ActualPosition')
            predicted_top3 = predictions.nsmallest(3, 'PredictedPosition')
            actual_top3_drivers = set(actual_top3['DriverName'])
            predicted_top3_drivers = set(predicted_top3['DriverName'])
            top3_acc = len(actual_top3_drivers & predicted_top3_drivers) / 3
            
            all_metrics.append({
                'Race': f"{event_name} (R{round_num})",
                'MAE': mae,
                'Exact': exact,
                'Within1': within1,
                'Within3': within3,
                'Top3': top3_acc,
                'N': len(top10_finishers)
            })
            
            # Display top 10 predictions
            print(f"\nPredicted Top 10:")
            print(f"{'Rank':<6} {'Driver':<12} {'Pred':<6} {'Actual':<8} {'Error':<8} {'Status':<10}")
            print("-" * 70)
            
            for idx, row in predictions.head(10).iterrows():
                driver = row['DriverName']
                pred_pos = row['PredictedPosition']
                actual_pos = row.get('ActualPosition', np.nan)
                
                if pd.notna(actual_pos):
                    error = abs(pred_pos - actual_pos)
                    if error == 0:
                        status = "Exact"
                    elif error <= 1:
                        status = "Close"
                    elif error <= 3:
                        status = "Fair"
                    else:
                        status = "Poor"
                    actual_str = f"{int(actual_pos)}"
                    error_str = f"{int(error)}"
                else:
                    status = "N/A"
                    actual_str = "N/A"
                    error_str = "N/A"
                
                print(f"{pred_pos:<6} {driver:<12} {pred_pos:<6} {actual_str:<8} {error_str:<8} {status:<10}")
            
            print(f"\nMetrics (Top 10 finishers only, N={len(top10_finishers)}):")
            print(f"  MAE: {mae:.2f} positions")
            print(f"  Exact: {exact*100:.1f}%")
            print(f"  Within 1: {within1*100:.1f}%")
            print(f"  Within 3: {within3*100:.1f}%")
            print(f"  Top-3 Accuracy: {top3_acc*100:.1f}%")
            
            # Check for backmarkers in top 10
            backmarker_teams = ['Haas', 'Williams', 'Sauber', 'Alpine']  # Adjust as needed
            top10_drivers = predictions.head(10)
            backmarkers_in_top10 = top10_drivers[top10_drivers['TeamName'].isin(backmarker_teams)]
            
            if len(backmarkers_in_top10) > 0:
                print(f"\n⚠️  WARNING: {len(backmarkers_in_top10)} backmarker(s) in predicted top 10:")
                for _, row in backmarkers_in_top10.iterrows():
                    print(f"    {row['DriverName']} ({row['TeamName']}) - Predicted: {row['PredictedPosition']}")
            else:
                print(f"\n✓ No backmarkers in predicted top 10")
            
            print()
            
        except Exception as e:
            print(f"  Error processing race: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    if len(all_metrics) > 0:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        summary_df = pd.DataFrame(all_metrics)
        print(f"\nAverage Metrics across {len(all_metrics)} races:")
        print(f"  MAE: {summary_df['MAE'].mean():.2f} positions")
        print(f"  Exact Accuracy: {summary_df['Exact'].mean()*100:.1f}%")
        print(f"  Within 1 Position: {summary_df['Within1'].mean()*100:.1f}%")
        print(f"  Within 3 Positions: {summary_df['Within3'].mean()*100:.1f}%")
        print(f"  Top-3 Accuracy: {summary_df['Top3'].mean()*100:.1f}%")
        
        print(f"\nPer-Race Metrics:")
        print(summary_df.to_string(index=False))

if __name__ == "__main__":
    test_predictions(num_races=10)

