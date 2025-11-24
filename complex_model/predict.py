"""
Complex F1 Prediction Model Inference.
Loads trained complex model and makes predictions with extensive feature engineering.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import sys

# Import complex feature preparation from train.py
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import importlib.util
spec = importlib.util.spec_from_file_location("train_module", parent_dir / "complex_model" / "train.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

prepare_complex_features = train_module.prepare_complex_features
ComplexF1NeuralNetwork = train_module.ComplexF1NeuralNetwork


class F1Dataset:
    """Dataset class for F1 features and labels."""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def handle_nan_values(X: np.ndarray) -> np.ndarray:
    """Handle NaN values in feature matrix."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        with np.errstate(all='ignore'):
            for i in range(X.shape[1]):
                nan_mask = np.isnan(X[:, i])
                if nan_mask.all():
                    X[:, i] = 0
                else:
                    non_nan_values = X[~nan_mask, i]
                    if len(non_nan_values) > 0:
                        fill_val = np.nanmean(non_nan_values)
                        if np.isnan(fill_val) or np.isinf(fill_val):
                            fill_val = np.nanmedian(non_nan_values)
                            if np.isnan(fill_val) or np.isinf(fill_val):
                                fill_val = 0
                        X[:, i] = np.nan_to_num(X[:, i], nan=fill_val)
                    else:
                        X[:, i] = 0
    return X


def load_model(model_dir: str = None):
    """Load trained complex model and scaler."""
    if model_dir is None:
        script_dir = Path(__file__).parent
        model_dir = script_dir.parent / 'models'
    else:
        model_dir = Path(model_dir)
    
    model_path = model_dir / 'f1_predictor_model_complex.pth'
    scaler_path = model_dir / 'scaler_complex.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please train the model first.")
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Determine input size from scaler
    input_size = scaler.n_features_in_
    
    # Initialize model (architecture must match training)
    model = ComplexF1NeuralNetwork(
        input_size=input_size,
        hidden_sizes=[512, 256, 128, 64],
        dropout_rate=0.3,
        equal_init=False
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, scaler


def predict_race(drivers_df: pd.DataFrame, model, scaler, training_df: pd.DataFrame = None):
    """
    Predict positions for all drivers in a race.
    
    Args:
        drivers_df: DataFrame with driver features
        model: Trained model
        scaler: Fitted scaler
        training_df: Optional training data for feature calculations
        
    Returns:
        DataFrame with predictions, sorted by predicted position
    """
    # Prepare complex features (this will calculate all features)
    X, y_dummy, _, _, feature_names = prepare_complex_features(
        drivers_df, historical_df=training_df, 
        filter_dnf=False, filter_outliers=False, top10_only=False
    )
    
    # Handle NaN values
    X = handle_nan_values(X)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        predictions = model(X_tensor).cpu().numpy()
        if predictions.ndim > 1:
            predictions = predictions.flatten()
    
    # Clip to valid range (1-10 for top 10 model)
    predictions = np.clip(predictions, 1, 10)
    
    # Add predictions to DataFrame
    result_df = drivers_df.copy()
    result_df['PredictedPosition'] = predictions
    
    # Sort by predicted position
    result_df = result_df.sort_values('PredictedPosition')
    result_df['Rank'] = range(1, len(result_df) + 1)
    
    return result_df


def main():
    """Main function for making predictions."""
    parser = argparse.ArgumentParser(description='F1 Position Prediction (Complex Model)')
    parser.add_argument('--input-file', type=str, help='CSV file with driver features')
    parser.add_argument('--model-dir', type=str, help='Directory containing model files')
    parser.add_argument('--output-file', type=str, help='Output CSV file for predictions')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading complex model...")
    try:
        model, scaler = load_model(args.model_dir)
        print(f"  Model loaded successfully")
        print(f"  Input features: {scaler.n_features_in_}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first by running: python complex_model/train.py")
        return
    
    # Load training data for feature calculations
    training_data_path = Path('data') / 'training_data.csv'
    training_df = None
    if training_data_path.exists():
        try:
            training_df = pd.read_csv(training_data_path)
            print(f"  Loaded training data: {len(training_df)} samples")
        except Exception:
            print("  Warning: Could not load training data for feature calculations")
    
    # Load input data
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return
        
        print(f"\nLoading input data from {input_path}...")
        drivers_df = pd.read_csv(input_path)
        print(f"  Found {len(drivers_df)} drivers")
        
        # Make predictions
        print("\nMaking predictions...")
        results = predict_race(drivers_df, model, scaler, training_df)
        
        # Display top 10
        top10 = results.head(10)
        print("\n" + "=" * 70)
        print("PREDICTED TOP 10")
        print("=" * 70)
        print(f"{'Rank':<6} {'Driver':<20} {'Predicted':<12} {'Season Points':<15}")
        print("-" * 70)
        for _, row in top10.iterrows():
            driver_name = row.get('DriverName', f"Driver {row.get('DriverNumber', '?')}")
            pred_pos = row['PredictedPosition']
            points = row.get('SeasonPoints', 0)
            print(f"{row['Rank']:<6} {driver_name:<20} {pred_pos:<12.2f} {points:<15.0f}")
        
        # Save results
        if args.output_file:
            output_path = Path(args.output_file)
            results.to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")
        else:
            # Save to default location
            output_path = Path('predictions_complex.csv')
            results.to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")
    else:
        print("\nNo input file provided. Use --input-file to specify a CSV with driver features.")
        print("\nExample usage:")
        print("  python complex_model/predict.py --input-file data/test_data.csv")


if __name__ == '__main__':
    main()

