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
from top10.feature_calculation import calculate_future_race_features
from top10.config import FEATURE_COLS
from top10.train import run_experiment
from top10.data_utils import load_f1_data, build_race_list, format_predictions

app = Flask(__name__)
CORS(app)

model = None
scaler = None
model_type = None
device = None
test_df = None
training_df = None


def init_model():
    global model, scaler, model_type, device
    model_dir = Path(__file__).parent.parent / 'models'
    model, scaler, model_type, device = load_model(str(model_dir), 'neural_network')
    print(f"Model loaded successfully! ({model_type})")
    print(f"Using device: {device}")


def load_data():
    global test_df, training_df
    data_dir = Path(__file__).parent.parent / 'data'
    test_df, training_df = load_f1_data(data_dir)


# Initialize on import
try:
    init_model()
    load_data()
except Exception as e:
    print(f"Startup error: {e}")
    print("API will not function properly until model and data are loaded.")


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_loaded': test_df is not None
    })


@app.route('/api/features', methods=['GET'])
def list_features():
    return jsonify({
        "all_features": FEATURE_COLS,
        "recommended": FEATURE_COLS,
    })


@app.route('/api/races', methods=['GET'])
def get_races():
    if test_df is None:
        return jsonify({'error': 'Test data not loaded'}), 500

    if not {'Year', 'EventName', 'RoundNumber'}.issubset(test_df.columns):
        return jsonify({'error': 'Required columns not found in test data'}), 500

    races_list = build_race_list(test_df)
    return jsonify({'races': races_list})


@app.route('/api/predict', methods=['POST'])
def predict():
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
        race_df = test_df[
            (test_df['Year'] == year) &
            (test_df['EventName'] == event_name) &
            (test_df['RoundNumber'] == round_number)
        ].copy()

        if race_df.empty:
            race_df = calculate_future_race_features(
                test_df, year, round_number, event_name, training_df
            )
            is_future = True
        else:
            is_future = False

        if race_df.empty:
            return jsonify({'error': 'No drivers found for this race'}), 404

        top10, all_drivers = predict_race_top10(race_df, model, scaler, model_type, device)
        predictions_filtered, predictions_unfiltered = format_predictions(top10, is_future)

        return jsonify({
            'race': {
                'year': year,
                'eventName': event_name,
                'roundNumber': round_number,
                'isFuture': is_future,
            },
            'predictionsFiltered': predictions_filtered,
            'predictionsUnfiltered': predictions_unfiltered,
            'totalDrivers': len(all_drivers),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/experiments', methods=['POST'])
def create_experiment():
    data = request.get_json(force=True) or {}
    features = data.get('features') or []
    max_epochs = int(data.get('max_epochs', 80))

    if not isinstance(features, list) or not features:
        return jsonify({"error": "features must be a non-empty list"}), 400

    try:
        metrics = run_experiment(features, max_epochs=max_epochs)
        return jsonify({
            "features": metrics.get("features", []),
            "metrics": {k: v for k, v in metrics.items() if k != "features"},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    try:
        init_model()
        load_data()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        print("Make sure models are trained and data files exist.")

    # host='0.0.0.0' so ngrok/other tunnels can reach the server.
    # When using ngrok, run: ngrok http 5000 --host-header=rewrite
    app.run(host='0.0.0.0', debug=True, port=5000)
