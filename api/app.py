"""
Flask API for F1 Race Prediction
Provides endpoints for race selection and predictions
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path to import top10 modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from top10.model_loader import load_model
from top10.predict import predict_race_top10
from top10.race_selection import get_future_races
from top10.feature_calculation import calculate_future_race_features

app = Flask(__name__)
CORS(app)

# Global variables for model and data
model = None
scaler = None
model_type = None
device = None
test_df = None
training_df = None

def load_data():
    """Load test and training data."""
    global test_df, training_df
    
    data_dir = Path(__file__).parent.parent / 'data'
    test_data_path = data_dir / 'test_data.csv'
    training_data_path = data_dir / 'training_data.csv'
    
    if test_data_path.exists():
        test_df = pd.read_csv(test_data_path)
        if 'EventName' in test_df.columns:
            test_df = test_df[~test_df['EventName'].str.contains('Pre-Season|Pre Season|Testing', case=False, na=False)].copy()
    
    if training_data_path.exists():
        try:
            training_df = pd.read_csv(training_data_path)
            if 'EventName' in training_df.columns:
                training_df = training_df[~training_df['EventName'].str.contains('Pre-Season|Pre Season|Testing', case=False, na=False)].copy()
        except Exception:
            training_df = None

def init_model():
    """Initialize the model and scaler."""
    global model, scaler, model_type, device
    
    model_dir = Path(__file__).parent.parent / 'models'
    try:
        model, scaler, model_type, device = load_model(str(model_dir), 'neural_network')
        print(f"Model loaded successfully! ({model_type})")
        print(f"Using device: {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Initialize on import
try:
    init_model()
    load_data()
except Exception as e:
    print(f"Startup error: {e}")
    print("API will not function properly until model and data are loaded.")

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_loaded': test_df is not None
    })

@app.route('/api/races', methods=['GET'])
def get_races():
    """Get list of available races."""
    if test_df is None:
        return jsonify({'error': 'Test data not loaded'}), 500
    
    if 'Year' not in test_df.columns or 'EventName' not in test_df.columns or 'RoundNumber' not in test_df.columns:
        return jsonify({'error': 'Required columns not found in test data'}), 500
    
    # Get unique races from test data
    completed_races = test_df[['Year', 'EventName', 'RoundNumber']].drop_duplicates()
    
    # Get future races for each year
    unique_years = sorted(test_df['Year'].unique())
    future_races_list = []
    
    for year in unique_years:
        try:
            future_races = get_future_races(year)
            if not future_races.empty:
                future_races_clean = future_races[['Year', 'EventName', 'RoundNumber']].drop_duplicates()
                # Only add future races that aren't already in completed races
                future_only = future_races_clean[
                    ~future_races_clean.set_index(['Year', 'EventName', 'RoundNumber']).index.isin(
                        completed_races.set_index(['Year', 'EventName', 'RoundNumber']).index
                    )
                ]
                future_races_list.append(future_only)
        except Exception:
            pass
    
    # Combine all races
    all_races = completed_races.copy()
    if future_races_list:
        all_future = pd.concat(future_races_list, ignore_index=True)
        all_races = pd.concat([all_races, all_future], ignore_index=True)
    
    all_races = all_races.sort_values(['Year', 'RoundNumber'], ascending=[False, False])
    
    # Check which races are future (no data in test_df)
    races_list = []
    for _, race in all_races.iterrows():
        has_data = not test_df[
            (test_df['Year'] == race['Year']) & 
            (test_df['EventName'] == race['EventName']) &
            (test_df['RoundNumber'] == race['RoundNumber'])
        ].empty
        
        races_list.append({
            'year': int(race['Year']),
            'eventName': race['EventName'],
            'roundNumber': int(race['RoundNumber']),
            'isFuture': not has_data
        })
    
    return jsonify({'races': races_list})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions for a specific race."""
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if test_df is None:
        return jsonify({'error': 'Test data not loaded'}), 500
    
    data = request.get_json()
    year = data.get('year')
    event_name = data.get('eventName')
    round_number = data.get('roundNumber')
    
    if not all([year, event_name, round_number]):
        return jsonify({'error': 'Missing required parameters: year, eventName, roundNumber'}), 400
    
    try:
        # Get race data
        race_df = test_df[
            (test_df['Year'] == year) & 
            (test_df['EventName'] == event_name) &
            (test_df['RoundNumber'] == round_number)
        ].copy()
        
        # If no data found, it's a future race - calculate features
        if race_df.empty:
            race_df = calculate_future_race_features(
                test_df, year, round_number, event_name, training_df
            )
            is_future = True
        else:
            is_future = False
        
        if race_df.empty:
            return jsonify({'error': 'No drivers found for this race'}), 404
        
        # Make predictions
        top10, all_drivers = predict_race_top10(race_df, model, scaler, model_type, device)
        
        # Determine filter reasons for each driver
        def get_filter_reason(row):
            """Determine why a driver was filtered out, if applicable.
            Only filters based on unpredictable events (DNF, large position drops),
            not based on finishing outside top 10 (which is just a training filter).
            Uses actual grid position from qualifying, not the average.
            """
            if is_future:
                return None  # Future races have no actual results
            
            actual_pos = row.get('ActualPosition')
            if pd.isna(actual_pos):
                return None
            
            actual_pos = int(actual_pos)
            
            # Use actual grid position from qualifying (stored in ActualGridPosition)
            # Fall back to average GridPosition if ActualGridPosition not available
            grid_pos = row.get('ActualGridPosition')
            if pd.isna(grid_pos):
                grid_pos = row.get('GridPosition')
            
            # Check for DNF/DSQ/DNS
            is_dnf = row.get('IsDNF', False)
            if pd.notna(is_dnf) and is_dnf:
                return 'DNF/DSQ'
            
            # Check for large position drop (finish > grid + 6)
            # This indicates an unpredictable event like a crash or damage
            if pd.notna(grid_pos):
                grid_pos = float(grid_pos)
                position_drop = actual_pos - grid_pos
                if position_drop > 6:
                    return f'Dropped {int(position_drop)} places (grid {int(grid_pos)} → finish {actual_pos})'
            
            return None
        
        # Format results - both filtered and unfiltered
        predictions_filtered = []
        predictions_unfiltered = []
        
        for _, row in top10.iterrows():
            has_constructor = 'Constructor' in row.index
            actual_pos = row.get('ActualPosition')
            actual_position = int(actual_pos) if pd.notna(actual_pos) else None
            
            # Get actual grid position from stored data (ActualGridPosition column)
            # This is the actual qualifying position, not the average
            actual_grid_pos = row.get('ActualGridPosition')
            if pd.isna(actual_grid_pos):
                # Fallback to average if actual not available
                actual_grid_pos = row.get('GridPosition')
            
            display_grid_pos = int(actual_grid_pos) if pd.notna(actual_grid_pos) else None
            
            filter_reason = get_filter_reason(row)
            is_filtered = filter_reason is not None
            
            pred_data = {
                'rank': int(row.get('Rank', 0)),
                'driverName': str(row.get('DriverName', 'Unknown')),
                'driverNumber': int(row.get('DriverNumber', 0)) if pd.notna(row.get('DriverNumber')) else None,
                'predictedPosition': float(row['PredictedPosition']),
                'actualPosition': actual_position,
                'gridPosition': display_grid_pos,
                'constructor': str(row.get('Constructor', 'Unknown')) if has_constructor else None,
                'filterReason': filter_reason,
                'isFiltered': is_filtered
            }
            
            # Add to both lists
            predictions_unfiltered.append(pred_data)
            if not is_filtered:
                predictions_filtered.append(pred_data)
        
        return jsonify({
            'race': {
                'year': year,
                'eventName': event_name,
                'roundNumber': round_number,
                'isFuture': is_future
            },
            'predictionsFiltered': predictions_filtered,
            'predictionsUnfiltered': predictions_unfiltered,
            'totalDrivers': len(all_drivers)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize on startup
    try:
        init_model()
        load_data()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        print("Make sure models are trained and data files exist.")
    
    app.run(debug=True, port=5000)

