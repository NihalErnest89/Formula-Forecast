"""
Comparison script to test model performance with ACTUAL vs AVERAGE grid positions.

This addresses the train/test mismatch:
- Current: Training uses REAL grid positions, future predictions use ESTIMATED averages
- Test: Training uses AVERAGE grid positions (same as future predictions)

We'll compare:
1. Model trained with ACTUAL GridPosition (current baseline)
2. Model trained with AVERAGE GridPosition (matches future prediction scenario)
3. Model trained WITHOUT GridPosition (baseline for comparison)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import from train.py
from train import F1Dataset, F1NeuralNetwork, evaluate_model, get_feature_importance


def calculate_average_grid_position(df: pd.DataFrame, driver_num: str, current_year: int, current_round: int) -> float:
    """
    Calculate average grid position for a driver using only data up to the current race.
    This simulates what we'd use for future race predictions.
    
    Args:
        df: DataFrame with all race data
        driver_num: Driver number as string
        current_year: Current race year
        current_round: Current race round number
        
    Returns:
        Average grid position for the driver
    """
    # Get all races for this driver BEFORE the current race
    driver_races = df[
        (df['DriverNumber'] == driver_num) &
        ((df['Year'] < current_year) | 
         ((df['Year'] == current_year) & (df['RoundNumber'] < current_round)))
    ]
    
    if driver_races.empty:
        return np.nan
    
    # Get grid positions (may be called GridPosition or other names)
    grid_positions = driver_races['GridPosition'].dropna()
    
    if len(grid_positions) > 0:
        return grid_positions.mean()
    else:
        return np.nan


def replace_with_avg_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace actual GridPosition with average grid position for each driver.
    This simulates training on the same type of data we'll have for future races.
    """
    df = df.copy()
    
    # Sort by year and round to process in order
    df = df.sort_values(['Year', 'RoundNumber'])
    
    # Create new column for average grid position
    df['AvgGridPosition'] = np.nan
    
    # Calculate average grid position for each race
    for idx, row in df.iterrows():
        driver_num = row['DriverNumber']
        year = row['Year']
        round_num = row['RoundNumber']
        
        avg_grid = calculate_average_grid_position(df, driver_num, year, round_num)
        df.at[idx, 'AvgGridPosition'] = avg_grid
    
    # Replace GridPosition with AvgGridPosition
    # For rows where AvgGridPosition is NaN (first race for a driver), use the actual GridPosition as fallback
    df['GridPosition'] = df['AvgGridPosition'].fillna(df['GridPosition'])
    
    return df


def prepare_features_and_labels(df: pd.DataFrame, use_grid: bool = True, use_avg_grid: bool = False):
    """
    Prepare feature matrix and label vector from DataFrame.
    
    Args:
        df: DataFrame with features and labels
        use_grid: If True, include GridPosition feature
        use_avg_grid: If True and use_grid is True, use AvgGridPosition instead of GridPosition
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    base_features = ['SeasonPoints', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                     'ConstructorPoints', 'ConstructorStanding', 'RecentForm']
    
    if use_grid:
        if use_avg_grid and 'AvgGridPosition' in df.columns:
            feature_cols = base_features + ['AvgGridPosition']
            # Rename for consistency
            df = df.copy()
            df['GridPosition'] = df['AvgGridPosition']
        else:
            feature_cols = base_features + ['GridPosition']
    else:
        feature_cols = base_features
    
    # Select features
    X = df[feature_cols].copy()
    
    # Handle missing values - use median for more robust handling
    # Also clip extreme values that might be outliers
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(median_val)
        
        # Clip extreme outliers (beyond 3 standard deviations) to reduce their impact
        # Skip GridPosition as it has specific valid range
        if col != 'GridPosition' and col != 'AvgGridPosition':
            mean_val = X[col].mean()
            std_val = X[col].std()
            if not pd.isna(std_val) and std_val > 0:
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Select labels (ActualPosition - finishing position 1-20)
    y = df['ActualPosition'].values
    
    # Remove any NaN positions (DNF, DSQ, etc.)
    valid_mask = ~pd.isna(y)
    X = X[valid_mask]
    y = y[valid_mask]
    
    return X.values, y, feature_cols


def train_model(X_train, y_train, X_val, y_val,
                epochs=100, batch_size=32, learning_rate=0.001, device='cpu'):
    """Train the neural network model."""
    # Create datasets and dataloaders
    train_dataset = F1Dataset(X_train, y_train)
    val_dataset = F1Dataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model with correct input size
    num_features = X_train.shape[1]
    model = F1NeuralNetwork(input_size=num_features, hidden_sizes=[128, 64, 32], 
                           dropout_rate=0.3, equal_init=True).to(device)
    
    # Loss function and optimizer
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=8, min_lr=1e-6)
    
    best_val_mae = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 20
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_total_loss = 0
        val_preds = []
        val_labels_list = []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device).float()
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_total_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_loss = val_total_loss / len(val_loader)
        val_mae_epoch = mean_absolute_error(val_labels_list, val_preds)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_mae_epoch < best_val_mae:
            best_val_mae = val_mae_epoch
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break
        
        if (epoch + 1) % 20 == 0:
            # Calculate train MAE for logging
            model.eval()
            train_preds = []
            train_labels_list = []
            with torch.no_grad():
                for features, labels in train_loader:
                    features, labels = features.to(device), labels.to(device).float()
                    outputs = model(features)
                    train_preds.extend(outputs.cpu().numpy())
                    train_labels_list.extend(labels.cpu().numpy())
            train_mae_epoch = mean_absolute_error(train_labels_list, train_preds)
            model.train()
            
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {avg_loss:.4f}, Train MAE: {train_mae_epoch:.3f} positions")
            print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae_epoch:.3f} positions")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\nEarly stopping at epoch {epoch + 1}")
    print(f"Best validation MAE: {best_val_mae:.3f} positions")
    
    return model


def main():
    """Compare model with actual vs average grid positions."""
    print("=" * 70)
    print("GRID POSITION: ACTUAL vs AVERAGE COMPARISON")
    print("=" * 70)
    print("Testing model performance:")
    print("  1. WITH ACTUAL GridPosition (current baseline)")
    print("  2. WITH AVERAGE GridPosition (matches future prediction scenario)")
    print("  3. WITHOUT GridPosition (baseline for comparison)")
    print()
    print("Why this matters:")
    print("  - Future predictions use AVERAGE grid positions (estimated)")
    print("  - Training with AVERAGE positions eliminates train/test mismatch")
    print("  - This shows if model can learn to work with estimates")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading data...")
    training_path = Path('data') / 'training_data.csv'
    test_path = Path('data') / 'test_data.csv'
    
    if not training_path.exists():
        raise FileNotFoundError(f"Training data not found at {training_path}. Run collect_data.py first.")
    
    training_df = pd.read_csv(training_path)
    test_df = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
    
    print(f"Training samples: {len(training_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Create version with average grid positions
    print("\nCalculating average grid positions for training data...")
    training_df_avg = replace_with_avg_grid(training_df)
    
    results = {}
    
    # Test three configurations
    configs = [
        (True, False, "WITH ACTUAL GridPosition"),
        (True, True, "WITH AVERAGE GridPosition"),
        (False, False, "WITHOUT GridPosition")
    ]
    
    for use_grid, use_avg_grid, config_name in configs:
        num_features = 7 if use_grid else 6
        
        print(f"\n{'='*70}")
        print(f"Training: {config_name} ({num_features} features)")
        print(f"{'='*70}")
        
        # Select appropriate dataframe
        if use_avg_grid:
            train_df = training_df_avg
        else:
            train_df = training_df
        
        # Prepare training data
        X_train, y_train, feature_names = prepare_features_and_labels(
            train_df, use_grid=use_grid, use_avg_grid=use_avg_grid
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        print(f"  Features shape: {X_train_scaled.shape}")
        print(f"  Features: {', '.join(feature_names)}")
        print(f"  Labels shape: {y_train.shape}")
        
        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42
        )
        
        # Train model
        model = train_model(
            X_train_split, y_train_split, 
            X_val_split, y_val_split,
            epochs=100,
            batch_size=32,
            learning_rate=0.001,
            device=device
        )
        
        # Evaluate on validation set
        val_dataset = F1Dataset(X_val_split, y_val_split)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        criterion = nn.HuberLoss(delta=1.0)
        
        val_loss, val_mae, val_rmse, val_r2, val_w1, val_w2, val_w3, val_preds, val_labels = evaluate_model(
            model, val_loader, criterion, device
        )
        
        print(f"\nValidation Metrics ({config_name}):")
        print(f"  MAE: {val_mae:.3f} positions")
        print(f"  RMSE: {val_rmse:.3f} positions")
        print(f"  R²: {val_r2:.4f}")
        print(f"  Within 1 position: {val_w1:.1f}%")
        print(f"  Within 2 positions: {val_w2:.1f}%")
        print(f"  Within 3 positions: {val_w3:.1f}%")
        
        # Evaluate on test set if available
        test_mae = test_rmse = test_r2 = test_w1 = test_w2 = test_w3 = None
        if not test_df.empty:
            # For test set, use actual grid positions (they're real for completed races)
            X_test, y_test, _ = prepare_features_and_labels(test_df, use_grid=use_grid, use_avg_grid=False)
            X_test_scaled = scaler.transform(X_test)
            
            test_dataset = F1Dataset(X_test_scaled, y_test)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            test_loss, test_mae, test_rmse, test_r2, test_w1, test_w2, test_w3, test_preds, test_labels = evaluate_model(
                model, test_loader, criterion, device
            )
            
            print(f"\nTest Metrics ({config_name}):")
            print(f"  MAE: {test_mae:.3f} positions")
            print(f"  RMSE: {test_rmse:.3f} positions")
            print(f"  R²: {test_r2:.4f}")
            print(f"  Within 1 position: {test_w1:.1f}%")
            print(f"  Within 2 positions: {test_w2:.1f}%")
            print(f"  Within 3 positions: {test_w3:.1f}%")
        
        # Feature importance
        feature_importances = get_feature_importance(model, feature_names, device)
        print(f"\nFeature Importances ({config_name}):")
        for name, importance in feature_importances.items():
            print(f"  {name}: {importance:.4f}")
        
        # Store results
        results[config_name] = {
            'num_features': num_features,
            'val_mae': float(val_mae),
            'val_rmse': float(val_rmse),
            'val_r2': float(val_r2),
            'val_within_1': float(val_w1),
            'val_within_2': float(val_w2),
            'val_within_3': float(val_w3),
            'test_mae': float(test_mae) if test_mae else None,
            'test_rmse': float(test_rmse) if test_rmse else None,
            'test_r2': float(test_r2) if test_r2 else None,
            'test_within_1': float(test_w1) if test_w1 else None,
            'test_within_2': float(test_w2) if test_w2 else None,
            'test_within_3': float(test_w3) if test_w3 else None,
            'feature_importances': {k: float(v) for k, v in feature_importances.items()}
        }
    
    # Print comparison summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    actual = results["WITH ACTUAL GridPosition"]
    avg = results["WITH AVERAGE GridPosition"]
    no_grid = results["WITHOUT GridPosition"]
    
    print(f"\nValidation Set Comparison:")
    print(f"{'Metric':<25} {'ACTUAL':<12} {'AVERAGE':<12} {'NO GRID':<12}")
    print("-" * 70)
    print(f"{'MAE (positions)':<25} {actual['val_mae']:<12.3f} {avg['val_mae']:<12.3f} {no_grid['val_mae']:<12.3f}")
    print(f"{'RMSE (positions)':<25} {actual['val_rmse']:<12.3f} {avg['val_rmse']:<12.3f} {no_grid['val_rmse']:<12.3f}")
    print(f"{'R²':<25} {actual['val_r2']:<12.4f} {avg['val_r2']:<12.4f} {no_grid['val_r2']:<12.4f}")
    print(f"{'Within 1 position (%)':<25} {actual['val_within_1']:<12.1f} {avg['val_within_1']:<12.1f} {no_grid['val_within_1']:<12.1f}")
    print(f"{'Within 2 positions (%)':<25} {actual['val_within_2']:<12.1f} {avg['val_within_2']:<12.1f} {no_grid['val_within_2']:<12.1f}")
    print(f"{'Within 3 positions (%)':<25} {actual['val_within_3']:<12.1f} {avg['val_within_3']:<12.1f} {no_grid['val_within_3']:<12.1f}")
    
    if actual['test_mae'] is not None:
        print(f"\nTest Set Comparison:")
        print(f"{'Metric':<25} {'ACTUAL':<12} {'AVERAGE':<12} {'NO GRID':<12}")
        print("-" * 70)
        print(f"{'MAE (positions)':<25} {actual['test_mae']:<12.3f} {avg['test_mae']:<12.3f} {no_grid['test_mae']:<12.3f}")
        print(f"{'RMSE (positions)':<25} {actual['test_rmse']:<12.3f} {avg['test_rmse']:<12.3f} {no_grid['test_rmse']:<12.3f}")
        print(f"{'R²':<25} {actual['test_r2']:<12.4f} {avg['test_r2']:<12.4f} {no_grid['test_r2']:<12.4f}")
        print(f"{'Within 1 position (%)':<25} {actual['test_within_1']:<12.1f} {avg['test_within_1']:<12.1f} {no_grid['test_within_1']:<12.1f}")
        print(f"{'Within 2 positions (%)':<25} {actual['test_within_2']:<12.1f} {avg['test_within_2']:<12.1f} {no_grid['test_within_2']:<12.1f}")
        print(f"{'Within 3 positions (%)':<25} {actual['test_within_3']:<12.1f} {avg['test_within_3']:<12.1f} {no_grid['test_within_3']:<12.1f}")
    
    # Determine winner
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    
    # Compare AVERAGE vs ACTUAL
    avg_vs_actual_diff = avg['val_mae'] - actual['val_mae']
    avg_vs_actual_pct = (avg_vs_actual_diff / actual['val_mae']) * 100
    
    print(f"\nAVERAGE vs ACTUAL GridPosition:")
    if avg_vs_actual_diff <= 0.1:  # Within 0.1 positions
        print(f"✓ Training with AVERAGE grid positions performs SIMILARLY to ACTUAL")
        print(f"  Difference: {avg_vs_actual_diff:.3f} positions ({avg_vs_actual_pct:.2f}%)")
        print(f"  → RECOMMENDED: Use AVERAGE grid positions for training")
        print(f"     This eliminates train/test mismatch while maintaining accuracy")
    elif avg_vs_actual_diff > 0:
        print(f"⚠ Training with AVERAGE grid positions performs WORSE than ACTUAL")
        print(f"  Degradation: {avg_vs_actual_diff:.3f} positions ({avg_vs_actual_pct:.2f}%)")
        print(f"  → Consider: Is the accuracy loss acceptable for better robustness?")
    else:
        print(f"✓ Training with AVERAGE grid positions performs BETTER than ACTUAL")
        print(f"  Improvement: {abs(avg_vs_actual_diff):.3f} positions ({abs(avg_vs_actual_pct):.2f}%)")
        print(f"  → RECOMMENDED: Use AVERAGE grid positions for training")
    
    # Compare AVERAGE vs NO GRID
    avg_vs_no_diff = avg['val_mae'] - no_grid['val_mae']
    avg_vs_no_pct = (avg_vs_no_diff / no_grid['val_mae']) * 100
    
    print(f"\nAVERAGE vs NO GridPosition:")
    if avg_vs_no_diff < 0:
        print(f"✓ Training with AVERAGE grid positions is BETTER than NO grid position")
        print(f"  Improvement: {abs(avg_vs_no_diff):.3f} positions ({abs(avg_vs_no_pct):.2f}%)")
        print(f"  → AVERAGE grid position is still valuable even when estimated")
    else:
        print(f"⚠ Training with AVERAGE grid positions is WORSE than NO grid position")
        print(f"  Degradation: {avg_vs_no_diff:.3f} positions ({avg_vs_no_pct:.2f}%)")
        print(f"  → Consider removing grid position if estimates are too noisy")
    
    if actual['test_mae'] is not None:
        test_avg_vs_actual = avg['test_mae'] - actual['test_mae']
        test_avg_vs_actual_pct = (test_avg_vs_actual / actual['test_mae']) * 100
        
        print(f"\nTest Set Analysis:")
        if abs(test_avg_vs_actual) <= 0.15:  # Within 0.15 positions
            print(f"✓ AVERAGE grid positions perform SIMILARLY on test set")
            print(f"  Difference: {test_avg_vs_actual:.3f} positions ({test_avg_vs_actual_pct:.2f}%)")
            print(f"  → Strong evidence that AVERAGE grid positions work well")
        else:
            print(f"  Test difference: {test_avg_vs_actual:.3f} positions ({test_avg_vs_actual_pct:.2f}%)")
    
    # Final recommendation
    print(f"\n{'='*70}")
    print("FINAL RECOMMENDATION")
    print(f"{'='*70}")
    
    if avg['val_mae'] <= actual['val_mae'] * 1.05 and avg['val_mae'] < no_grid['val_mae']:
        print("→ Use AVERAGE GridPosition for training")
        print("  - Eliminates train/test mismatch")
        print("  - Maintains good accuracy")
        print("  - Better than removing grid position entirely")
    elif avg['val_mae'] > actual['val_mae'] * 1.10:
        print("→ Consider removing GridPosition")
        print("  - Average estimates may be too noisy")
        print("  - Accuracy loss from estimates may not be worth it")
    else:
        print("→ Use AVERAGE GridPosition with caution")
        print("  - Small accuracy trade-off for better robustness")
        print("  - Monitor performance on future race predictions")
    
    # Save results
    json_dir = Path('json')
    json_dir.mkdir(exist_ok=True)
    results_path = json_dir / 'avg_grid_comparison.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

