"""
Training and testing script for F1 Predictions using Random Forest.
Trains a Random Forest model on past 5 seasons and tests on 2025 season data.
This is a traditional ML approach for comparison with the neural network.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def load_data(data_dir: str = 'data'):
    """
    Load training and test data.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (training_df, test_df, metadata)
    """
    training_path = Path(data_dir) / 'training_data.csv'
    test_path = Path(data_dir) / 'test_data.csv'
    metadata_path = Path(data_dir) / 'metadata.json'
    
    if not training_path.exists():
        raise FileNotFoundError(f"Training data not found at {training_path}. Run collect_data.py first.")
    
    training_df = pd.read_csv(training_path)
    test_df = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
    
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    return training_df, test_df, metadata


def prepare_features_and_labels(df: pd.DataFrame):
    """
    Prepare feature matrix and label vector from DataFrame.
    
    Args:
        df: DataFrame with features and labels
        
    Returns:
        Tuple of (X, y) where X is feature matrix and y is label vector (finishing positions)
    """
    feature_cols = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition', 
                   'ConstructorPoints', 'ConstructorStanding', 'GridPosition']
    
    # Select features
    X = df[feature_cols].copy()
    
    # Handle missing values - use median for more robust handling
    # Fill NaN with median (more robust than mean for skewed data)
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                # If median is also NaN, fill with 0
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(median_val)
    
    # Select labels (ActualPosition - finishing position 1-20)
    y = df['ActualPosition'].values
    
    # Remove any NaN positions (DNF, DSQ, etc.)
    valid_mask = ~pd.isna(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    return X, y


def train_model(X_train: np.ndarray, y_train: np.ndarray, model_type: str = 'random_forest'):
    """
    Train a regression model to predict finishing positions (1-20).
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels (finishing positions)
        model_type: Type of model ('random_forest' or 'gradient_boosting')
        
    Returns:
        Trained model and scaler
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Choose model - using REGRESSION models to predict positions
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    print(f"Training {model_type} regression model to predict finishing positions...")
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def evaluate_model(model, scaler, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate model performance for position prediction (regression).
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        X_test: Test feature matrix
        y_test: Test labels (actual positions)
        driver_mapping: Optional mapping from driver number to name
        
    Returns:
        Dictionary with evaluation metrics
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Position accuracy (within 1, 2, 3 positions)
    position_error = np.abs(y_test - y_pred)
    within_1 = np.mean(position_error <= 1) * 100
    within_2 = np.mean(position_error <= 2) * 100
    within_3 = np.mean(position_error <= 3) * 100
    
    print(f"\nModel Evaluation (Position Prediction):")
    print(f"  Mean Absolute Error (MAE): {mae:.3f} positions")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.3f} positions")
    print(f"  R-squared (R²): {r2:.4f}")
    print(f"\nPosition Accuracy:")
    print(f"  Within 1 position: {within_1:.1f}%")
    print(f"  Within 2 positions: {within_2:.1f}%")
    print(f"  Within 3 positions: {within_3:.1f}%")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_names = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                         'ConstructorPoints', 'ConstructorStanding', 'GridPosition']
        importances = model.feature_importances_
        print(f"\nFeature Importances (Weight Distribution):")
        for name, importance in zip(feature_names, importances):
            print(f"  {name}: {importance:.4f}")
    
    # Plot prediction vs actual
    try:
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Position')
        plt.ylabel('Predicted Position')
        plt.title('Predicted vs Actual Finishing Positions (Random Forest)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # Create images directory if it doesn't exist
        images_dir = Path('images')
        images_dir.mkdir(exist_ok=True)
        
        scatter_path = images_dir / 'prediction_scatter_rf.png'
        plt.savefig(scatter_path, dpi=150)
        print(f"\nPrediction scatter plot saved to {scatter_path}")
        plt.close()
    except Exception as e:
        print(f"Could not save scatter plot: {e}")
    
    feature_names = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                     'ConstructorPoints', 'ConstructorStanding', 'GridPosition']
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'within_1_position': within_1,
        'within_2_positions': within_2,
        'within_3_positions': within_3,
        'predictions': y_pred,
        'feature_importances': dict(zip(feature_names, importances)) if hasattr(model, 'feature_importances_') else None
    }


def save_model(model, scaler, output_dir: str = 'models'):
    """
    Save trained model and scaler.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        output_dir: Directory to save model files
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    model_path = Path(output_dir) / 'f1_predictor_model_rf.pkl'
    scaler_path = Path(output_dir) / 'scaler_rf.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


def main():
    """Main function to train and test the F1 prediction model."""
    print("F1 Position Prediction Model Training (Random Forest)")
    print("=" * 60)
    print("Note: Random Forest considers all features equally at start,")
    print("      then learns optimal feature importance through training.")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    training_df, test_df, _ = load_data()
    
    print(f"Training samples: {len(training_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Prepare training data
    print("\nPreparing training data...")
    X_train, y_train = prepare_features_and_labels(training_df)
    print(f"  Features shape: {X_train.shape}")
    print(f"  Labels shape: {y_train.shape}")
    print(f"  Position range: {y_train.min():.0f} - {y_train.max():.0f}")
    
    # Split training data for validation
    # Note: No stratification for regression (continuous target)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train model
    model, scaler = train_model(X_train_split, y_train_split, model_type='random_forest')
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_results = evaluate_model(model, scaler, X_val_split, y_val_split)
    
    # Evaluate on test set if available
    if not test_df.empty:
        print("\nEvaluating on test set...")
        X_test, y_test = prepare_features_and_labels(test_df)
        test_results = evaluate_model(model, scaler, X_test, y_test)
    else:
        print("\nNo test data available. Skipping test evaluation.")
        test_results = None
    
    # Save model
    save_model(model, scaler)
    
    # Save results - convert all numpy types to native Python types for JSON serialization
    feature_importances = val_results.get('feature_importances')
    if feature_importances:
        feature_importances = {k: float(v) for k, v in feature_importances.items()}
    
    results = {
        'model_type': 'random_forest_regression',
        'validation_mae': float(val_results['mae']),
        'validation_rmse': float(val_results['rmse']),
        'validation_r2': float(val_results['r2']),
        'validation_within_1': float(val_results['within_1_position']),
        'validation_within_2': float(val_results['within_2_positions']),
        'validation_within_3': float(val_results['within_3_positions']),
        'test_mae': float(test_results['mae']) if test_results else None,
        'test_rmse': float(test_results['rmse']) if test_results else None,
        'test_r2': float(test_results['r2']) if test_results else None,
        'test_within_1': float(test_results['within_1_position']) if test_results else None,
        'test_within_2': float(test_results['within_2_positions']) if test_results else None,
        'test_within_3': float(test_results['within_3_positions']) if test_results else None,
        'feature_importances': feature_importances
    }
    
    # Create json directory if it doesn't exist
    json_dir = Path('json')
    json_dir.mkdir(exist_ok=True)
    
    results_path = json_dir / 'training_results_rf.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Results saved to {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

