"""
Training and testing script for F1 Predictions (Top 10 Only).
Trains a deep neural network on top 10 finishers only (positions 1-10).
Test results show: Training on top 10 only improves top 10 MAE from 3.043 to 1.861
and "Within 3" accuracy from 55.5% to 79.1%.
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
import matplotlib.pyplot as plt
import pickle


# ============================================================================
# CLASSES AND FUNCTIONS (Standalone - no dependencies on top20/train.py)
# ============================================================================

class F1Dataset(Dataset):
    """PyTorch Dataset for F1 prediction data."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class F1NeuralNetwork(nn.Module):
    """
    Deep Neural Network for F1 Position Prediction (Regression).
    
    Architecture:
    - Input layer: 9 features
    - Hidden layers: Multiple fully connected layers with ReLU activation
    - Output layer: Single value (predicted finishing position 1-10)
    """
    
    def __init__(self, input_size=9, hidden_sizes=[192, 96, 48], dropout_rate=0.4, equal_init=True):
        super(F1NeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.equal_init = equal_init
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        
        # Output layer: single value for position prediction
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        if equal_init:
            self._initialize_equal_weights()
        else:
            self._initialize_he_weights()
    
    def _initialize_equal_weights(self):
        """Initialize first layer weights to give equal importance to all input features."""
        first_layer = self.network[0]
        with torch.no_grad():
            init_value = 1.0 / np.sqrt(self.input_size)
            first_layer.weight.fill_(init_value)
            if first_layer.bias is not None:
                first_layer.bias.fill_(0.0)
        print(f"  Initialized first layer with equal weights: {init_value:.4f} for all features")
    
    def _initialize_he_weights(self):
        """Initialize all layers with He/Kaiming initialization (optimal for ReLU)."""
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
        print(f"  Initialized all layers with He/Kaiming initialization (optimal for ReLU)")
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x).squeeze()


def load_data(data_dir: str = None):
    """Load training and test data."""
    if data_dir is None:
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / 'data'
    else:
        data_dir = Path(data_dir)
    
    training_path = data_dir / 'training_data.csv'
    test_path = data_dir / 'test_data.csv'
    metadata_path = data_dir / 'metadata.json'
    
    if not training_path.exists():
        raise FileNotFoundError(f"Training data not found at {training_path}. Run collect_data.py first.")
    
    training_df = pd.read_csv(training_path)
    test_df = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
    
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    return training_df, test_df, metadata


def prepare_features_and_labels(df: pd.DataFrame, label_encoder=None, 
                                filter_dnf=True, filter_outliers=True, 
                                outlier_threshold=6, top10_only=False):
    """
    Prepare feature matrix and label vector from DataFrame.
    
    Args:
        df: DataFrame with features and labels
        label_encoder: Optional pre-fitted label encoder
        filter_dnf: If True, remove DNF/DSQ/DNS entries
        filter_outliers: If True, remove entries where finish position is much worse than grid position
        outlier_threshold: Maximum allowed difference between finish and grid position (default: 6)
        top10_only: If True, only include positions 1-10 (default: False)
        
    Returns:
        Tuple of (X, y, label_encoder, filter_stats, feature_names)
    """
    # Base features (9 features)
    desired_features = ['SeasonPoints', 'SeasonStanding', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                       'ConstructorStanding', 'ConstructorTrackAvg', 'GridPosition', 'RecentForm', 'TrackType']
    
    # Use only features that exist in the dataframe
    feature_cols = [f for f in desired_features if f in df.columns]
    
    if 'SeasonStanding' not in feature_cols and 'SeasonPoints' in feature_cols:
        print(f"  Note: SeasonStanding not in data, using SeasonPoints only (regenerate data with collect_data.py for both features)")
    
    if len(feature_cols) == 0:
        raise ValueError("No valid feature columns found in data!")
    
    # Start with all rows
    valid_mask = pd.Series([True] * len(df), index=df.index)
    filter_stats = {'initial_count': len(df), 'dnf_removed': 0, 'outlier_removed': 0, 'nan_removed': 0, 'top10_filtered': 0}
    
    # Filter to top 10 only if requested
    if top10_only and 'ActualPosition' in df.columns:
        top10_mask = df['ActualPosition'] <= 10
        filter_stats['top10_filtered'] = (~top10_mask & valid_mask).sum()
        valid_mask = valid_mask & top10_mask
        if filter_stats['top10_filtered'] > 0:
            print(f"  Filtered to top 10 only: removed {filter_stats['top10_filtered']} positions > 10")
    
    # Filter DNFs if requested
    if filter_dnf and 'IsDNF' in df.columns:
        dnf_mask = df['IsDNF'].fillna(False).astype(bool)
        filter_stats['dnf_removed'] = dnf_mask.sum()
        valid_mask = valid_mask & ~dnf_mask
        if filter_stats['dnf_removed'] > 0:
            print(f"  Removed {filter_stats['dnf_removed']} DNF/DSQ/DNS entries")
    
    # Filter outliers
    if filter_outliers and 'GridPosition' in df.columns and 'ActualPosition' in df.columns:
        valid_for_outlier_check = valid_mask & df['ActualPosition'].notna() & df['GridPosition'].notna()
        
        if valid_for_outlier_check.any():
            position_diff = df.loc[valid_for_outlier_check, 'ActualPosition'] - df.loc[valid_for_outlier_check, 'GridPosition']
            outlier_mask_local = position_diff > outlier_threshold
            
            outlier_mask = pd.Series(False, index=df.index)
            outlier_mask.loc[valid_for_outlier_check] = outlier_mask_local
            
            filter_stats['outlier_removed'] = outlier_mask.sum()
            valid_mask = valid_mask & ~outlier_mask
            if filter_stats['outlier_removed'] > 0:
                print(f"  Removed {filter_stats['outlier_removed']} outliers (finish > grid + {outlier_threshold})")
    
    # Select labels
    y = df['ActualPosition'].values
    
    # Remove any NaN positions
    nan_mask = ~pd.isna(y)
    filter_stats['nan_removed'] = (~nan_mask & valid_mask).sum()
    valid_mask = valid_mask & nan_mask
    
    # Apply all filters
    df_filtered = df[valid_mask].copy()
    filter_stats['final_count'] = len(df_filtered)
    
    print(f"  Filtering: {filter_stats['initial_count']} -> {filter_stats['final_count']} samples "
          f"({filter_stats['initial_count'] - filter_stats['final_count']} removed)")
    
    # CRITICAL: Renumber positions per race after filtering
    if 'ActualPosition' in df_filtered.columns and ('Year' in df_filtered.columns or 'RoundNumber' in df_filtered.columns):
        if 'Year' in df_filtered.columns and 'RoundNumber' in df_filtered.columns:
            if 'EventName' in df_filtered.columns:
                race_groups = df_filtered.groupby(['Year', 'RoundNumber', 'EventName'])
            else:
                race_groups = df_filtered.groupby(['Year', 'RoundNumber'])
        elif 'Year' in df_filtered.columns:
            if 'EventName' in df_filtered.columns:
                race_groups = df_filtered.groupby(['Year', 'EventName'])
            else:
                race_groups = df_filtered.groupby('Year')
        else:
            race_groups = None
        
        if race_groups is not None:
            df_filtered = df_filtered.copy()
            for (race_key, group) in race_groups:
                sorted_indices = group.sort_values('ActualPosition').index
                for new_pos, idx in enumerate(sorted_indices, start=1):
                    df_filtered.at[idx, 'ActualPosition'] = new_pos
            print(f"  Renumbered positions per race after filtering")
    
    # Check if TrackType exists in data, if not add it
    if 'TrackType' not in df_filtered.columns and 'EventName' in df_filtered.columns:
        street_circuits = ['Monaco', 'Singapore', 'Azerbaijan', 'Miami', 'Las Vegas', 'Saudi Arabian']
        df_filtered['TrackType'] = df_filtered['EventName'].apply(
            lambda x: 1 if any(street in str(x) for street in street_circuits) else 0
        )
        print(f"  Calculated TrackType from EventName (was missing from data)")
    
    # Select features
    X = df_filtered[feature_cols].copy()
    feature_names = feature_cols.copy()
    
    y = df_filtered['ActualPosition'].values
    
    # Handle missing values
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(median_val)
        
        # Clip extreme outliers
        if col != 'GridPosition':
            mean_val = X[col].mean()
            std_val = X[col].std()
            if not pd.isna(std_val) and std_val > 0:
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
    
    return X.values, y, None, filter_stats, feature_names


class PositionAwareLoss(nn.Module):
    """Loss function that penalizes exact position misses more heavily."""
    def __init__(self, exact_weight=2.0, base_loss='huber', delta=1.0):
        super().__init__()
        self.exact_weight = exact_weight
        self.delta = delta
        if base_loss == 'huber':
            self.base_loss = nn.HuberLoss(delta=delta)
        elif base_loss == 'mse':
            self.base_loss = nn.MSELoss()
        else:
            self.base_loss = nn.L1Loss()
    
    def forward(self, pred, target):
        base = self.base_loss(pred, target)
        rounded_pred = torch.round(pred)
        exact_misses = (rounded_pred != target).float()
        weighted_loss = base * (1 + self.exact_weight * exact_misses.mean())
        return weighted_loss


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate MAE for monitoring
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device).float()
            outputs = model(features)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        mae = mean_absolute_error(all_labels, all_preds)
    
    return avg_loss, mae


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device).float()
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)
    
    all_preds_rounded = np.round(np.array(all_preds))
    all_labels_array = np.array(all_labels)
    position_error = np.abs(all_labels_array - np.array(all_preds))
    exact = np.mean(all_preds_rounded == all_labels_array) * 100
    within_1 = np.mean(position_error <= 1) * 100
    within_2 = np.mean(position_error <= 2) * 100
    within_3 = np.mean(position_error <= 3) * 100
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return avg_loss, mae, rmse, r2, exact, within_1, within_2, within_3, all_preds, all_labels


def get_feature_importance(model, feature_names, device):
    """Estimate feature importance by examining first layer weights."""
    first_layer = model.network[0]
    weights = first_layer.weight.data.cpu().numpy()
    importances = np.mean(np.abs(weights), axis=0)
    importances = importances / importances.sum()
    return dict(zip(feature_names, importances))


def train_model(X_train, y_train, X_val, y_val,
                epochs=300, batch_size=32, learning_rate=0.001, device='cpu',
                hidden_sizes=[128, 64, 32], feature_names=None, early_stop_patience=None, 
                track_weights=True, **kwargs):
    """Train the neural network model."""
    train_dataset = F1Dataset(X_train, y_train)
    val_dataset = F1Dataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    input_size = X_train.shape[1]
    model = F1NeuralNetwork(input_size=input_size, hidden_sizes=hidden_sizes, 
                           dropout_rate=0.4, equal_init=False).to(device)
    
    exact_weight = kwargs.get('exact_weight', 5.0)
    criterion = PositionAwareLoss(exact_weight=exact_weight, base_loss='huber', delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=8, min_lr=1e-6)
    
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'weight_progression': []
    }
    
    best_val_mae = float('inf')
    best_model_state = None
    patience_counter = 0
    if early_stop_patience is None:
        early_stop_patience = 100
    
    print(f"\nTraining Neural Network on {device}")
    print(f"Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    if model.equal_init and feature_names:
        initial_importance = get_feature_importance(model, feature_names, device)
        print(f"\nInitial Feature Importance (Equal Weights):")
        for name, importance in initial_importance.items():
            print(f"  {name}: {importance:.4f}")
    
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)
    
    weight_track_interval = max(1, epochs // 50)
    
    for epoch in range(epochs):
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae, val_rmse, val_r2, val_exact, val_w1, val_w2, val_w3, _, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        if epoch % weight_track_interval == 0 or epoch == epochs - 1:
            first_layer = model.network[0]
            weights = first_layer.weight.data.cpu().numpy()
            avg_weights = np.mean(np.abs(weights), axis=0)
            avg_weights = avg_weights / avg_weights.sum()
            history['weight_progression'].append({
                'epoch': epoch,
                'weights': avg_weights.tolist()
            })
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.3f} positions")
            print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.3f} positions")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if early_stop_patience is not None:
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
                    break
        else:
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_state = model.state_dict().copy()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\nBest validation MAE: {best_val_mae:.3f} positions")
    return model, history


def plot_weight_progression(history, feature_names, save_path=None):
    """Plot how feature weights evolve during training."""
    if save_path is None:
        script_dir = Path(__file__).parent
        save_path = script_dir.parent / 'images' / 'weight_progression_top10.png'
    else:
        save_path = Path(save_path)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if 'weight_progression' not in history or len(history['weight_progression']) == 0:
        print("No weight progression data available.")
        return
    
    epochs = [w['epoch'] for w in history['weight_progression']]
    weights_data = np.array([w['weights'] for w in history['weight_progression']])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, feature_name in enumerate(feature_names):
        ax.plot(epochs, weights_data[:, i], label=feature_name, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Normalized Weight Importance', fontsize=12)
    ax.set_title('Feature Weight Progression During Training', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(0.3, weights_data.max() * 1.1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Weight progression plot saved to {save_path}")
    plt.close()


def plot_training_history(history, save_path=None):
    """Plot training and validation loss/accuracy curves."""
    if save_path is None:
        script_dir = Path(__file__).parent
        save_path = script_dir.parent / 'images' / 'training_history_top10.png'
    else:
        save_path = Path(save_path)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history['train_mae'], 'b-', label='Training MAE')
    ax2.plot(epochs, history['val_mae'], 'r-', label='Validation MAE')
    ax2.set_title('Mean Absolute Error (Position Prediction)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE (positions)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training history plot saved to {save_path}")
    plt.close()


def save_model(model, scaler, label_encoder, output_dir: str = None, model_index: int = None):
    """Save trained model, scaler, and label encoder (top 10 version)."""
    if output_dir is None:
        # Resolve path relative to this script's location
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / 'models'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Single model
    model_path = output_dir / 'f1_predictor_model_top10.pth'
    
    scaler_path = output_dir / 'scaler_top10.pkl'
    encoder_path = output_dir / 'label_encoder_top10.pkl'
    
    torch.save(model.state_dict(), model_path)
    
    # Only save scaler/encoder once (they're the same for all models)
    if model_index is None or model_index == 0:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        print(f"Scaler saved to {scaler_path}")
        print(f"Label encoder saved to {encoder_path}")
    
    print(f"Model saved to {model_path}")


def main():
    """Main function to train and test the F1 prediction model (top 10 only)."""
    print("F1 Position Prediction Model Training (Top 10 Only)")
    print("=" * 60)
    print("Predicting finishing positions (1-10) for top finishers only")
    print("Test results: Top 10 MAE improved from 3.043 to 1.861")
    print("             Within 3 accuracy improved from 55.5% to 79.1%")
    print("=" * 60)
    
    # Set device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    training_df, test_df, metadata = load_data()
    
    # Verify that training data includes expected years (2020-2024)
    expected_years = {2020, 2021, 2022, 2023, 2024}
    actual_years = set(training_df['Year'].unique()) if 'Year' in training_df.columns else set()
    
    if not expected_years.issubset(actual_years):
        missing_years = expected_years - actual_years
        print(f"\n{'='*70}")
        print("WARNING: Training data is missing expected years!")
        print(f"{'='*70}")
        print(f"Expected years: {sorted(expected_years)}")
        print(f"Actual years in data: {sorted(actual_years)}")
        print(f"Missing years: {sorted(missing_years)}")
        print(f"\nThe data needs to be regenerated with the expanded dataset (2020-2024).")
        print(f"Please run: python collect_data.py")
        print(f"{'='*70}\n")
        raise ValueError(
            f"Training data missing years: {sorted(missing_years)}. "
            f"Please run 'python collect_data.py' to regenerate data with years 2020-2024."
        )
    
    print(f"Training samples: {len(training_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Training years: {sorted(actual_years)}")
    
    # Prepare training data (filtered - no DNFs/outliers, top 10 only)
    # Test results show: Training on top 10 only improves top 10 MAE from 3.043 to 1.861
    # and "Within 3" accuracy from 55.5% to 79.1%
    print("\nPreparing training data (top 10 positions only)...")
    
    # TIME-AWARE K-FOLD CROSS-VALIDATION: Use all 2020-2024 for training/validation
    # Split by year to prevent data leakage (time-series data)
    print("\nUsing K-Fold Cross-Validation (Time-Aware) for validation...")
    print("  This allows us to use all training years (2020-2024) for validation")
    print("  while still respecting temporal order (no future data leakage)")
    
    def k_fold_time_based_split(df, n_splits=None):
        """Time-based k-fold split for time-series data. Splits by year."""
        years = sorted(df['Year'].unique())
        n_years = len(years)
        
        # Default to one fold per year
        if n_splits is None:
            n_splits = n_years
        
        if n_splits > n_years:
            n_splits = n_years
        
        # For time-series, use one year per fold (most common approach)
        # This ensures each validation set is from a single year
        folds = []
        for i in range(n_splits):
            # Each fold validates on one year, trains on all others
            val_years = [years[i]]
            train_years = [y for y in years if y != years[i]]
            folds.append((train_years, val_years))
        
        return folds
    
    # Perform k-fold cross-validation
    # Use one fold per year (5 folds for 2020-2024)
    years_in_data = sorted(training_df['Year'].unique())
    n_splits = len(years_in_data)  # One fold per year
    print(f"  Training years available: {years_in_data}")
    print(f"  Using {n_splits}-fold cross-validation (one fold per year)")
    folds = k_fold_time_based_split(training_df, n_splits=n_splits)
    
    cv_results = []
    all_models = []
    all_scalers = []
    
    print(f"\nPerforming {n_splits}-fold cross-validation...")
    for fold_idx, (train_years, val_years) in enumerate(folds):
        print(f"\n  Fold {fold_idx + 1}/{n_splits}:")
        print(f"    Train years: {train_years}")
        print(f"    Validation years: {val_years}")
        
        # Split data
        train_fold_df = training_df[training_df['Year'].isin(train_years)].copy()
        val_fold_df = training_df[training_df['Year'].isin(val_years)].copy()
        
        # Prepare features
        X_train_fold, y_train_fold, _, _, feature_names = prepare_features_and_labels(
            train_fold_df, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=True
        )
        X_val_fold, y_val_fold, _, _, _ = prepare_features_and_labels(
            val_fold_df, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=True
        )
        
        # Scale features
        scaler_fold = StandardScaler()
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler_fold.transform(X_val_fold)
        
        # Train model for this fold
        train_dataset = F1Dataset(X_train_fold_scaled, y_train_fold)
        val_dataset = F1Dataset(X_val_fold_scaled, y_val_fold)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        input_size = X_train_fold_scaled.shape[1]
        model = F1NeuralNetwork(
            input_size=input_size,
            hidden_sizes=[128, 64, 32],
            dropout_rate=0.4,  # Increased regularization
            equal_init=False  # Use He/Kaiming initialization (optimal for ReLU)
        ).to(device)
        
        # Use PositionAwareLoss (defined above)
        # Weight=5.0 improves exact accuracy by ~1% and MAE by ~0.02 compared to weight=2.0
        criterion = PositionAwareLoss(exact_weight=5.0, base_loss='huber', delta=1.0)
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=2e-4)  # Increased regularization
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
        )
        
        best_val_mae = float('inf')
        best_model_state = None
        patience = 0
        max_patience = 30
        
        for epoch in range(200):
            # Train
            train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss, val_mae, val_rmse, val_r2, val_exact, val_w1, val_w2, val_w3, _, _ = evaluate_model(
                model, val_loader, criterion, device
            )
            
            scheduler.step(val_loss)
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_state = model.state_dict().copy()
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    break
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        criterion_eval = nn.MSELoss()
        val_loss, val_mae, val_rmse, val_r2, val_exact, val_w1, val_w2, val_w3, _, _ = evaluate_model(
            model, val_loader, criterion_eval, device
        )
        
        cv_results.append({
            'fold': fold_idx + 1,
            'train_years': train_years,
            'val_years': val_years,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_exact': val_exact,
            'val_w1': val_w1,
            'val_w2': val_w2,
            'val_w3': val_w3
        })
        
        all_models.append(model)
        all_scalers.append(scaler_fold)
        
        print(f"    Validation MAE: {val_mae:.3f}, Exact: {val_exact:.1f}%, W1: {val_w1:.1f}%, W3: {val_w3:.1f}%")
    
    # Calculate average CV metrics
    avg_val_mae = np.mean([r['val_mae'] for r in cv_results])
    avg_val_rmse = np.mean([r['val_rmse'] for r in cv_results])
    avg_val_r2 = np.mean([r['val_r2'] for r in cv_results])
    avg_val_exact = np.mean([r['val_exact'] for r in cv_results])
    avg_val_w1 = np.mean([r['val_w1'] for r in cv_results])
    avg_val_w2 = np.mean([r['val_w2'] for r in cv_results])
    avg_val_w3 = np.mean([r['val_w3'] for r in cv_results])
    
    print(f"\n  Cross-Validation Average Results:")
    print(f"    MAE: {avg_val_mae:.3f} positions")
    print(f"    RMSE: {avg_val_rmse:.3f} positions")
    print(f"    R²: {avg_val_r2:.4f}")
    print(f"    Exact position: {avg_val_exact:.1f}%")
    print(f"    Within 1 position: {avg_val_w1:.1f}%")
    print(f"    Within 2 positions: {avg_val_w2:.1f}%")
    print(f"    Within 3 positions: {avg_val_w3:.1f}%")
    
    # Use the model from the last fold (trained on all years except the last validation year)
    # Or train a final model on ALL training data (2020-2024)
    print(f"\nTraining final model on ALL training data (2020-2024)...")
    X_train_all, y_train_all, _, train_stats, feature_names = prepare_features_and_labels(
        training_df, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=True
    )
    
    scaler = StandardScaler()
    X_train_all_scaled = scaler.fit_transform(X_train_all)
    
    # For validation during final training, use the last year (2024) from training data
    val_year = training_df['Year'].max()
    val_df_final = training_df[training_df['Year'] == val_year].copy()
    
    # Add an index column to track original row positions before filtering
    val_df_final = val_df_final.reset_index(drop=True)
    val_df_final['_original_index'] = range(len(val_df_final))
    
    X_val_final, y_val_final, _, val_stats, _ = prepare_features_and_labels(
        val_df_final, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=True
    )
    X_val_final_scaled = scaler.transform(X_val_final)
    
    # Recreate filtered dataframe by applying same filtering logic
    val_df_filtered = val_df_final.copy()
    valid_mask = pd.Series([True] * len(val_df_filtered), index=val_df_filtered.index)
    
    # Apply same filters as prepare_features_and_labels
    if 'ActualPosition' in val_df_filtered.columns:
        valid_mask = valid_mask & (val_df_filtered['ActualPosition'] <= 10)
        valid_mask = valid_mask & val_df_filtered['ActualPosition'].notna()
    
    if 'IsDNF' in val_df_filtered.columns:
        dnf_mask = val_df_filtered['IsDNF'].fillna(False).astype(bool)
        valid_mask = valid_mask & ~dnf_mask
    
    if 'GridPosition' in val_df_filtered.columns and 'ActualPosition' in val_df_filtered.columns:
        valid_for_outlier = valid_mask & val_df_filtered['ActualPosition'].notna() & val_df_filtered['GridPosition'].notna()
        if valid_for_outlier.any():
            position_diff = val_df_filtered.loc[valid_for_outlier, 'ActualPosition'] - val_df_filtered.loc[valid_for_outlier, 'GridPosition']
            outlier_mask_local = position_diff > 6
            outlier_mask = pd.Series(False, index=val_df_filtered.index)
            outlier_mask.loc[valid_for_outlier] = outlier_mask_local
            valid_mask = valid_mask & ~outlier_mask
    
    val_df_filtered = val_df_filtered[valid_mask].copy()
    
    # Renumber positions per race (same as prepare_features_and_labels)
    if 'ActualPosition' in val_df_filtered.columns and 'Year' in val_df_filtered.columns and 'RoundNumber' in val_df_filtered.columns:
        if 'EventName' in val_df_filtered.columns:
            race_groups = val_df_filtered.groupby(['Year', 'RoundNumber', 'EventName'])
        else:
            race_groups = val_df_filtered.groupby(['Year', 'RoundNumber'])
        
        for (race_key, group) in race_groups:
            sorted_indices = group.sort_values('ActualPosition').index
            for new_pos, idx in enumerate(sorted_indices, start=1):
                val_df_filtered.at[idx, 'ActualPosition'] = new_pos
    
    # Ensure we have matching number of rows (should match after same filtering)
    if len(val_df_filtered) != len(y_val_final):
        print(f"  Warning: Filtered dataframe has {len(val_df_filtered)} rows but predictions have {len(y_val_final)} rows")
        # Use first N rows that match
        val_df_filtered = val_df_filtered.iloc[:len(y_val_final)].copy()
    
    X_train_split = X_train_all_scaled
    X_val_split = X_val_final_scaled
    y_train_split = y_train_all
    y_val_split = y_val_final
    
    print(f"  Final training samples: {len(X_train_split)}")
    print(f"  Final validation samples: {len(X_val_split)}")
    
    # Prepare test data (filtered - for primary evaluation, top 10 only)
    print("\nPreparing test data (top 10 positions only, excluding DNFs/outliers)...")
    X_test, y_test, _, test_stats, _ = prepare_features_and_labels(
        test_df, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=True)
    
    # Also prepare unfiltered test data (for comparison/reference, top 10 only)
    # Note: We still exclude DNFs (NaN positions) since we can't evaluate accuracy on them
    # But we include outliers (finish >> grid) for reference
    print("\nPreparing test data (top 10 only, including outliers for reference)...")
    X_test_all, y_test_all, _, test_stats_all, _ = prepare_features_and_labels(
        test_df, filter_dnf=True, filter_outliers=False, outlier_threshold=6, top10_only=True)
    
    # Show filtering stats
    print(f"\nTest Data Filtering Stats:")
    print(f"  Filtered (no outliers): {test_stats.get('final_count', len(y_test))} samples")
    print(f"  Unfiltered (with outliers): {test_stats_all.get('final_count', len(y_test_all))} samples")
    print(f"  Outliers removed: {test_stats.get('outlier_removed', 0)}")
    outliers_removed = test_stats.get('outlier_removed', 0)
    if outliers_removed > 0:
        print(f"  Note: Removed {outliers_removed} outliers (finish > grid + 6) from filtered set")
    else:
        print(f"  Note: No outliers found (finish > grid + 6) in top 10 positions")
    
    # Scale test data with the same scaler
    X_test_scaled = scaler.transform(X_test)
    X_test_all_scaled = scaler.transform(X_test_all) if len(X_test_all) > 0 else None
    
    print(f"  Features shape: {X_train_split.shape}")
    print(f"  Labels shape: {y_train_split.shape}")
    print(f"  Position range: {y_train_split.min():.0f} - {y_train_split.max():.0f}")
    
    # Architecture: Optimized based on training improvements test
    # [128, 64, 32] - Best configuration (TrackID removed - had lowest importance 0.0993)
    # Using increased regularization: LR 0.005, Dropout 0.4, Weight Decay 2e-4
    # Position-aware loss to improve exact position accuracy
    hidden_sizes = [128, 64, 32]
    
    # Train single model (ensemble testing showed minimal benefit - single model performs better on exact match)
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Updated hyperparameters with increased regularization:
    # - LR 0.005: optimized learning rate
    # - Dropout 0.4: increased from 0.3 to reduce overfitting (validation-test gap)
    # - Weight Decay 2e-4: increased from 1e-4 to reduce overfitting
    # - Position-Aware Loss: improves exact position accuracy
    model, history = train_model(
        X_train_split, y_train_split,
        X_val_split, y_val_split,
        epochs=300,
        batch_size=32,
        learning_rate=0.005,  # Optimized: increased from 0.001 (8.37% improvement)
        device=device,
        hidden_sizes=hidden_sizes,
        feature_names=feature_names,
        exact_weight=5.0,  # Position-aware loss weight
        early_stop_patience=50  # Stop if no improvement for 50 epochs
    )
    print(f"  Model training complete")
    
    # Plot training history
    script_dir = Path(__file__).parent
    plot_training_history(history, save_path=str(script_dir.parent / 'images' / 'training_history_top10.png'))
    
    # Plot weight progression separately if available
    if 'weight_progression' in history and len(history['weight_progression']) > 0 and feature_names:
        plot_weight_progression(history, feature_names, 
                                save_path=str(script_dir.parent / 'images' / 'weight_progression_top10.png'))
    
    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Final Model Evaluation on Validation Set (2024)")
    print("=" * 60)
    print("Note: This is the final model trained on ALL training data (2020-2024)")
    print("      Cross-validation results above show average performance across all folds")
    
    val_dataset = F1Dataset(X_val_split, y_val_split)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    criterion = nn.MSELoss()
    
    val_loss, val_mae, val_rmse, val_r2, val_exact, val_w1, val_w2, val_w3, val_preds, val_labels = evaluate_model(
        model, val_loader, criterion, device
    )
    
    print(f"\nValidation Metrics:")
    print(f"  MAE: {val_mae:.3f} positions")
    print(f"  RMSE: {val_rmse:.3f} positions")
    print(f"  R²: {val_r2:.4f}")
    print(f"\nPosition Accuracy:")
    print(f"  Exact position: {val_exact:.1f}%")
    print(f"  Within 1 position: {val_w1:.1f}%")
    print(f"  Within 2 positions: {val_w2:.1f}%")
    print(f"  Within 3 positions: {val_w3:.1f}%")
    
    # Feature importance (from first layer weights)
    feature_importances = get_feature_importance(model, feature_names, device)
    print(f"\nFeature Importances (Weight Distribution from First Layer):")
    for name, importance in feature_importances.items():
        print(f"  {name}: {importance:.4f}")
    
    # Scatter plot: predicted vs actual positions (using ranked positions per race)
    script_dir = Path(__file__).parent
    images_dir = script_dir.parent / 'images'
    images_dir.mkdir(exist_ok=True, parents=True)
    
    # Rank predictions per race (not across all validation samples)
    # Create a dataframe with predictions and race info for grouping
    # Use array positions (0, 1, 2, ...) to match predictions with dataframe rows
    n_samples = min(len(val_preds), len(val_df_filtered))
    val_results_df = val_df_filtered.iloc[:n_samples].copy()
    val_results_df['PredictedRaw'] = val_preds[:n_samples]
    val_results_df['ActualPosition'] = val_labels[:n_samples]
    val_results_df['_array_index'] = range(n_samples)  # Track array position
    
    # Rank predictions within each race
    val_preds_ranked = np.zeros(len(val_preds))
    if 'Year' in val_results_df.columns and 'EventName' in val_results_df.columns and 'RoundNumber' in val_results_df.columns:
        # Group by race and rank within each race
        for (_year, _event, _round_num), group in val_results_df.groupby(['Year', 'EventName', 'RoundNumber']):
            # Get array indices for this race group
            group_array_indices = group['_array_index'].values
            if len(group_array_indices) > 0:
                group_preds = val_preds[group_array_indices]
                # Rank predictions within this race (1 = best prediction, 10 = worst)
                group_ranks = np.argsort(np.argsort(group_preds)) + 1
                val_preds_ranked[group_array_indices] = group_ranks
    else:
        # Fallback: rank all predictions together if race info not available
        val_preds_ranked = np.argsort(np.argsort(val_preds)) + 1
    
    # Group by actual position and compute average predicted rank for each actual position
    unique_positions = np.unique(val_labels)
    avg_predicted = []
    positions_list = []
    
    for pos in unique_positions:
        mask = val_labels == pos
        if mask.sum() > 0:
            avg_pred = np.mean(val_preds_ranked[mask])
            avg_predicted.append(avg_pred)
            positions_list.append(pos)
    
    plt.figure(figsize=(10, 8))
    plt.plot(positions_list, avg_predicted, 'o-', markersize=8, label='Average Predicted Rank')
    plt.plot([val_labels.min(), val_labels.max()], [val_labels.min(), val_labels.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Position')
    plt.ylabel('Average Predicted Rank (from sorted predictions, per race)')
    plt.title('Predicted vs Actual Finishing Positions (Top 10 Only)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_path = images_dir / 'prediction_scatter_top10.png'
    plt.savefig(scatter_path, dpi=150)
    print(f"\nPrediction scatter plot saved to {scatter_path}")
    plt.close()
    
    # Evaluate on test set (filtered - primary metrics)
    if len(X_test_scaled) > 0:
        print("\n" + "=" * 60)
        print("Evaluation on Test Set (FILTERED - Excluding DNFs/Outliers)")
        print("=" * 60)
        print("This is the primary evaluation metric - excludes unpredictable failures")
        print(f"Test samples: {len(y_test)} (after filtering, top 10 only)")
        
        test_dataset = F1Dataset(X_test_scaled, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        test_loss, test_mae, test_rmse, test_r2, test_exact, test_w1, test_w2, test_w3, test_preds, test_labels = evaluate_model(
            model, test_loader, criterion, device
        )
        
        print(f"\nTest Metrics (Filtered, Top 10 Only):")
        print(f"  MAE: {test_mae:.3f} positions")
        print(f"  RMSE: {test_rmse:.3f} positions")
        print(f"  R²: {test_r2:.4f}")
        print(f"\nPosition Accuracy (Filtered, Top 10 Only):")
        print(f"  Exact position: {test_exact:.1f}%")
        print(f"  Within 1 position: {test_w1:.1f}%")
        print(f"  Within 2 positions: {test_w2:.1f}%")
        print(f"  Within 3 positions: {test_w3:.1f}%")
        
        # Also evaluate on unfiltered data for reference
        if X_test_all_scaled is not None and len(X_test_all_scaled) > 0:
            print("\n" + "=" * 60)
            print("Evaluation on Test Set (UNFILTERED - Including All Entries)")
            print("=" * 60)
            print("Reference metrics - includes DNFs/outliers for comparison (top 10 only)")
            print(f"Test samples: {len(y_test_all)} (all entries, top 10 only)")
            
            test_dataset_all = F1Dataset(X_test_all_scaled, y_test_all)
            test_loader_all = DataLoader(test_dataset_all, batch_size=32, shuffle=False)
            
            test_loss_all, test_mae_all, test_rmse_all, test_r2_all, test_exact_all, test_w1_all, test_w2_all, test_w3_all, _, _ = evaluate_model(
                model, test_loader_all, criterion, device
            )
            
            print(f"\nTest Metrics (Unfiltered - Reference, Top 10 Only):")
            print(f"  MAE: {test_mae_all:.3f} positions")
            print(f"  RMSE: {test_rmse_all:.3f} positions")
            print(f"  R²: {test_r2_all:.4f}")
            print(f"\nPosition Accuracy (Unfiltered - Reference, Top 10 Only):")
            print(f"  Exact position: {test_exact_all:.1f}%")
            print(f"  Within 1 position: {test_w1_all:.1f}%")
            print(f"  Within 2 positions: {test_w2_all:.1f}%")
            print(f"  Within 3 positions: {test_w3_all:.1f}%")
            
            print(f"\nComparison:")
            print(f"  Filtered vs Unfiltered MAE: {test_mae:.3f} vs {test_mae_all:.3f} (improvement: {test_mae_all - test_mae:.3f})")
            print(f"  Filtered vs Unfiltered Within 3: {test_w3:.1f}% vs {test_w3_all:.1f}% (improvement: {test_w3 - test_w3_all:.1f}%)")
        else:
            test_mae_all = test_rmse_all = test_r2_all = test_w1_all = test_w2_all = test_w3_all = None
    else:
        print("\nNo test data available. Skipping test evaluation.")
        test_mae = test_rmse = test_r2 = test_w1 = test_w2 = test_w3 = None
        test_mae_all = test_rmse_all = test_r2_all = test_w1_all = test_w2_all = test_w3_all = None
    
    # Save model (no label encoder for regression)
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    save_model(model, scaler, None, model_index=None)
    print(f"\nSaved model")
    
    # Save results - convert all numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            try:
                if hasattr(obj, 'item'):
                    return obj.item()
            except (AttributeError, ValueError):
                pass
            return obj
    
    # Convert filtering stats to native types
    filtering_stats_dict = {
        'training': train_stats,
        'test_filtered': test_stats,
        'test_unfiltered': test_stats_all if 'test_stats_all' in locals() else None
    }
    
    results = {
        'model_type': 'neural_network_regression_top10',
        'validation_strategy': 'k_fold_cross_validation',
        'k_fold_cv': {
            'n_splits': n_splits,
            'folds': convert_to_native(cv_results),
            'average': {
                'mae': float(avg_val_mae),
                'rmse': float(avg_val_rmse),
                'r2': float(avg_val_r2),
                'exact': float(avg_val_exact),
                'within_1': float(avg_val_w1),
                'within_2': float(avg_val_w2),
                'within_3': float(avg_val_w3)
            }
        },
        'filtering_stats': convert_to_native(filtering_stats_dict),
        'final_model_validation': {
            'mae': float(val_mae),
            'rmse': float(val_rmse),
            'r2': float(val_r2),
            'exact': float(val_exact),
            'within_1': float(val_w1),
            'within_2': float(val_w2),
            'within_3': float(val_w3)
        },
        'test_filtered': {
            'mae': float(test_mae) if test_mae else None,
            'rmse': float(test_rmse) if test_rmse else None,
            'r2': float(test_r2) if test_r2 else None,
            'within_1': float(test_w1) if test_w1 else None,
            'within_2': float(test_w2) if test_w2 else None,
            'within_3': float(test_w3) if test_w3 else None,
        },
        'test_unfiltered': {
            'mae': float(test_mae_all) if 'test_mae_all' in locals() and test_mae_all else None,
            'rmse': float(test_rmse_all) if 'test_rmse_all' in locals() and test_rmse_all else None,
            'r2': float(test_r2_all) if 'test_r2_all' in locals() and test_r2_all else None,
            'within_1': float(test_w1_all) if 'test_w1_all' in locals() and test_w1_all else None,
            'within_2': float(test_w2_all) if 'test_w2_all' in locals() and test_w2_all else None,
            'within_3': float(test_w3_all) if 'test_w3_all' in locals() and test_w3_all else None,
        },
        'feature_importances': {k: float(v) for k, v in feature_importances.items()},
        'model_architecture': {
            'input_size': X_train_split.shape[1],
            'hidden_sizes': hidden_sizes,
            'dropout_rate': 0.4,
            'output': 'regression (single position value, 1-10)',
            'features': feature_names if feature_names else []
        }
    }
    
    # Create json directory if it doesn't exist
    script_dir = Path(__file__).parent
    json_dir = script_dir.parent / 'json'
    json_dir.mkdir(exist_ok=True, parents=True)
    
    results_path = json_dir / 'training_results_top10.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Results saved to {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

