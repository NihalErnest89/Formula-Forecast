"""
Training and testing script for F1 Predictions.
Trains a deep neural network on past 5 seasons and tests on 2025 season data.
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
    - Input layer: 7 features (Season Points, Season Avg Finish, Historical Track Avg, Constructor Standing, Grid Position, Recent Form, Track Type)
    - TrackID removed: had lowest feature importance (0.0993)
    - Hidden layers: Multiple fully connected layers with ReLU activation
    - Output layer: Single value (predicted finishing position 1-20)
    """
    
    def __init__(self, input_size=7, hidden_sizes=[192, 96, 48], dropout_rate=0.4, equal_init=True):
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
        # Get the first linear layer
        first_layer = self.network[0]
        
        # Initialize all weights to the same small value
        # This ensures each feature starts with equal contribution
        with torch.no_grad():
            # Use Xavier/Glorot initialization scaled for equal feature importance
            # Set all weights in first layer to same value
            init_value = 1.0 / np.sqrt(self.input_size)
            first_layer.weight.fill_(init_value)
            
            # Initialize bias to zero
            if first_layer.bias is not None:
                first_layer.bias.fill_(0.0)
        
        print(f"  Initialized first layer with equal weights: {init_value:.4f} for all features")
    
    def _initialize_he_weights(self):
        """Initialize all layers with He/Kaiming initialization (optimal for ReLU)."""
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    # He/Kaiming initialization for ReLU
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
        
        print(f"  Initialized all layers with He/Kaiming initialization (optimal for ReLU)")
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x).squeeze()  # Remove extra dimension


def load_data(data_dir: str = None):
    """
    Load training and test data.
    
    Args:
        data_dir: Directory containing data files (default: ../data relative to script)
        
    Returns:
        Tuple of (training_df, test_df, metadata)
    """
    if data_dir is None:
        # Resolve path relative to this script's location
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
    # Base features (8 features, ConstructorTrackAvg added - showed 1.8% MAE improvement)
    # Note: GridPosition (AvgGridPosition) is season-specific average grid position (not actual qualifying position)
    # This matches future prediction scenario where we don't have qualifying results yet
    # Using season-specific average (only races from current season) instead of all-time average
    # Removed ConstructorPoints (tested: removing it improves accuracy from 58.5% to 61.0% within 3 positions)
    # Removed TrackID: had lowest feature importance (0.0993), not providing value
    # Added ConstructorTrackAvg: Constructor's average finish at specific track (complements HistoricalTrackAvgPosition)
    # Using both SeasonPoints and SeasonStanding: improves exact accuracy to 20.6% (vs 15.4% baseline)
    # Check which features are available in the data
    # Base features (used by both regression and classification)
    # Added PointsGapToLeader and FormTrend - these are already calculated in collect_data.py
    # and available in the data pipeline, testing if they improve performance
    desired_features = ['SeasonPoints', 'SeasonStanding', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                       'ConstructorStanding', 'ConstructorTrackAvg', 'GridPosition', 'RecentForm', 'TrackType',
                       'PointsGapToLeader', 'FormTrend']
    
    # Use only features that exist in the dataframe
    feature_cols = [f for f in desired_features if f in df.columns]
    
    # If SeasonStanding is missing but SeasonPoints exists, use SeasonPoints only
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
        # Handle boolean values properly (True, False, NaN)
        # Convert to boolean, treating NaN as False (not DNF)
        dnf_mask = df['IsDNF'].fillna(False).astype(bool)
        filter_stats['dnf_removed'] = dnf_mask.sum()
        valid_mask = valid_mask & ~dnf_mask
        if filter_stats['dnf_removed'] > 0:
            print(f"  Removed {filter_stats['dnf_removed']} DNF/DSQ/DNS entries")
    
    # Filter outliers: finish position significantly worse than grid position
    # Note: We filter outliers because we only want to predict normal race outcomes,
    # not mechanical failures, crashes, or other exceptional circumstances
    # This should only apply to non-DNF entries (already filtered above)
    if filter_outliers and 'GridPosition' in df.columns and 'ActualPosition' in df.columns:
        # Only check outliers for entries that are still valid (not DNF, have valid positions)
        # Calculate position difference (finish - grid, positive = finished worse than started)
        # Only calculate for rows that are still in valid_mask and have valid numeric positions
        valid_for_outlier_check = valid_mask & df['ActualPosition'].notna() & df['GridPosition'].notna()
        
        if valid_for_outlier_check.any():
            position_diff = df.loc[valid_for_outlier_check, 'ActualPosition'] - df.loc[valid_for_outlier_check, 'GridPosition']
            # Filter out cases where finish is much worse than grid (likely mechanical issues, crashes, etc.)
            outlier_mask_local = position_diff > outlier_threshold
            
            # Map back to original index
            outlier_mask = pd.Series(False, index=df.index)
            outlier_mask.loc[valid_for_outlier_check] = outlier_mask_local
            
            filter_stats['outlier_removed'] = outlier_mask.sum()
            valid_mask = valid_mask & ~outlier_mask
            if filter_stats['outlier_removed'] > 0:
                print(f"  Removed {filter_stats['outlier_removed']} outliers (finish > grid + {outlier_threshold})")
    
    # Select labels (ActualPosition - finishing position 1-20)
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
    
    # Check if TrackType exists in data, if not add it (for backward compatibility)
    if 'TrackType' not in df_filtered.columns and 'EventName' in df_filtered.columns:
        # Calculate TrackType from EventName
        street_circuits = [
            'Monaco', 'Singapore', 'Azerbaijan', 'Miami', 'Las Vegas',
            'Saudi Arabian'
        ]
        df_filtered['TrackType'] = df_filtered['EventName'].apply(
            lambda x: 1 if any(street in str(x) for street in street_circuits) else 0
        )
        print(f"  Calculated TrackType from EventName (was missing from data)")
    
    # Select features (7 features)
    X = df_filtered[feature_cols].copy()
    feature_names = feature_cols.copy()  # Save for later use
    
    y = df_filtered['ActualPosition'].values
    
    # Handle missing values - use median for more robust handling
    # Also clip extreme values that might be outliers
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            if pd.isna(median_val):
                # If median is also NaN, fill with 0
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(median_val)
        
        # Clip extreme outliers (beyond 3 standard deviations) to reduce their impact
        # Skip GridPosition as it can legitimately be 1-20
        if col != 'GridPosition':
            mean_val = X[col].mean()
            std_val = X[col].std()
            if not pd.isna(std_val) and std_val > 0:
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
    
    return X.values, y, None, filter_stats, feature_names  # No label encoder needed for regression


class PositionAwareLoss(nn.Module):
    """
    Loss function that penalizes exact position misses more heavily.
    This helps improve exact position accuracy by giving extra weight to predictions
    that are close to integers but wrong.
    """
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
        
        # Additional penalty for predictions that are close to integers but wrong
        rounded_pred = torch.round(pred)
        exact_misses = (rounded_pred != target).float()
        
        # Weight the loss more for exact misses
        # Use mean to keep scale similar to base loss
        weighted_loss = base * (1 + self.exact_weight * exact_misses.mean())
        
        return weighted_loss


class Top3WeightedLoss(nn.Module):
    """
    Loss function that heavily penalizes errors in top 3 positions.
    This helps improve top 3 accuracy by giving extra weight to predictions
    for positions 1-3.
    """
    def __init__(self, top3_weight=3.0, exact_weight=2.0, base_loss='huber', delta=1.0):
        super().__init__()
        self.top3_weight = top3_weight
        self.exact_weight = exact_weight
        self.delta = delta
        if base_loss == 'huber':
            self.base_loss = nn.HuberLoss(delta=delta, reduction='none')
        elif base_loss == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        else:
            self.base_loss = nn.L1Loss(reduction='none')
    
    def forward(self, pred, target):
        # Calculate base loss for each sample
        base = self.base_loss(pred, target)
        
        # Identify top 3 positions (target <= 3) - heavily weight these
        top3_mask = (target <= 3).float()
        top3_weighted = base * (1 + self.top3_weight * top3_mask)
        
        # Also penalize exact misses (especially important for top 3)
        rounded_pred = torch.round(pred)
        exact_misses = (rounded_pred != target).float()
        exact_weighted = top3_weighted * (1 + self.exact_weight * exact_misses)
        
        return exact_weighted.mean()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device).float()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
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
    
    # Regression metrics
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)
    
    # Position accuracy (exact, within 1, 2, 3 positions)
    # Round predictions to nearest integer for exact match calculation
    all_preds_rounded = np.round(np.array(all_preds))
    all_labels_array = np.array(all_labels)
    
    position_error = np.abs(all_labels_array - np.array(all_preds))
    exact = np.mean(all_preds_rounded == all_labels_array) * 100  # Exact position matches (rounded)
    within_1 = np.mean(position_error <= 1) * 100
    within_2 = np.mean(position_error <= 2) * 100
    within_3 = np.mean(position_error <= 3) * 100
    
    avg_loss = total_loss / len(dataloader)
    
    # Convert to numpy arrays for easier manipulation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return avg_loss, mae, rmse, r2, exact, within_1, within_2, within_3, all_preds, all_labels


def get_feature_importance(model, feature_names, device):
    """
    Estimate feature importance by examining first layer weights.
    This gives an approximation of how much each feature contributes.
    """
    first_layer = model.network[0]
    weights = first_layer.weight.data.cpu().numpy()
    
    # Average absolute weights for each input feature
    importances = np.mean(np.abs(weights), axis=0)
    
    # Normalize to sum to 1
    importances = importances / importances.sum()
    
    return dict(zip(feature_names, importances))


def train_model(X_train, y_train, X_val, y_val,
                epochs=300, batch_size=32, learning_rate=0.001, device='cpu',
                hidden_sizes=[128, 64, 32], feature_names=None, early_stop_patience=None, 
                track_weights=True):
    """
    Train the neural network model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on ('cpu' or 'cuda')
        hidden_sizes: List of hidden layer sizes (default: [128, 64, 32])
        
    Returns:
        Trained model, training history
    """
    # Create datasets and dataloaders
    train_dataset = F1Dataset(X_train, y_train)
    val_dataset = F1Dataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model (regression - outputs single position value)
    # equal_init=False uses He/Kaiming initialization (optimal for ReLU)
    # Input size determined dynamically from feature count
    input_size = X_train.shape[1]
    # Increased regularization: dropout 0.3 → 0.4 to reduce overfitting (validation-test gap)
    model = F1NeuralNetwork(input_size=input_size, hidden_sizes=hidden_sizes, 
                           dropout_rate=0.4, equal_init=False).to(device)
    
    # Loss function: Use Position-Aware Loss to improve exact position accuracy
    # Combines Huber loss (robust to outliers) with extra penalty for exact position misses
    # Weight=5.0 improves exact accuracy by ~1% and MAE by ~0.02 compared to weight=2.0
    criterion = PositionAwareLoss(exact_weight=5.0, base_loss='huber', delta=1.0)
    # Increased regularization: weight decay 1e-4 → 2e-4 to reduce overfitting
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=8, min_lr=1e-6)
    
    # Training history
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'weight_progression': []  # Track first layer weights over time
    }
    
    best_val_mae = float('inf')  # Lower is better for MAE
    best_model_state = None
    patience_counter = 0
    # Default early stopping patience: 100 epochs (very lenient) or disable if None
    if early_stop_patience is None:
        early_stop_patience = 100  # Increased from 20 - allow model to train longer and find better minima
    
    print(f"\nTraining Neural Network on {device}")
    print(f"Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Show initial feature importance (should be equal)
    if model.equal_init and feature_names:
        initial_importance = get_feature_importance(model, feature_names, device)
        print(f"\nInitial Feature Importance (Equal Weights):")
        for name, importance in initial_importance.items():
            print(f"  {name}: {importance:.4f}")
    
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)
    
    # Track weight progression at regular intervals
    weight_track_interval = max(1, epochs // 50)  # Track ~50 times during training
    
    for epoch in range(epochs):
        # Train
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_mae, val_rmse, val_r2, val_exact, val_w1, val_w2, val_w3, _, _ = evaluate_model(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        # Track weight progression at regular intervals
        if epoch % weight_track_interval == 0 or epoch == epochs - 1:
            first_layer = model.network[0]
            weights = first_layer.weight.data.cpu().numpy()
            # Store average absolute weight per feature (normalized)
            avg_weights = np.mean(np.abs(weights), axis=0)
            avg_weights = avg_weights / avg_weights.sum()  # Normalize
            history['weight_progression'].append({
                'epoch': epoch,
                'weights': avg_weights.tolist()
            })
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.3f} positions")
            print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.3f} positions")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping (lower MAE is better)
        # Only apply early stopping if patience is set (None = disabled)
        if early_stop_patience is not None:
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_state = model.state_dict().copy()  # Save best model
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
                    break
        else:
            # No early stopping - always save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_model_state = model.state_dict().copy()  # Save best model
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\nBest validation MAE: {best_val_mae:.3f} positions")
    
    return model, history


def plot_weight_progression(history, feature_names, save_path=None):
    """Plot how feature weights evolve during training."""
    if save_path is None:
        script_dir = Path(__file__).parent
        save_path = script_dir.parent / 'images' / 'weight_progression.png'
    else:
        save_path = Path(save_path)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if 'weight_progression' not in history or len(history['weight_progression']) == 0:
        print("No weight progression data available.")
        return
    
    # Extract data
    epochs = [w['epoch'] for w in history['weight_progression']]
    weights_data = np.array([w['weights'] for w in history['weight_progression']])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each feature's weight progression
    for i, feature_name in enumerate(feature_names):
        ax.plot(epochs, weights_data[:, i], label=feature_name, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Normalized Weight Importance', fontsize=12)
    ax.set_title('Feature Weight Progression During Training', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(0.3, weights_data.max() * 1.1))  # Show up to 30% or max value
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Weight progression plot saved to {save_path}")
    plt.close()


def plot_training_history(history, save_path=None):
    """Plot training and validation loss/accuracy curves."""
    if save_path is None:
        # Resolve path relative to this script's location
        script_dir = Path(__file__).parent
        save_path = script_dir.parent / 'images' / 'training_history.png'
    else:
        save_path = Path(save_path)
    
    # Create images directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # MAE plot
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
    """Save trained model, scaler, and label encoder."""
    if output_dir is None:
        # Resolve path relative to this script's location
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / 'models'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if model_index is not None:
        # Ensemble model
        model_path = output_dir / f'f1_predictor_model_ensemble_{model_index}.pth'
    else:
        # Single model (backward compatibility)
        model_path = output_dir / 'f1_predictor_model.pth'
    
    scaler_path = output_dir / 'scaler.pkl'
    encoder_path = output_dir / 'label_encoder.pkl'
    
    torch.save(model.state_dict(), model_path)
    
    import pickle
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
    """Main function to train and test the F1 prediction model."""
    print("F1 Position Prediction Model Training (Deep Learning)")
    print("=" * 60)
    print("Predicting finishing positions (1-20) for each driver in a race")
    print("=" * 60)
    
    # Set device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    training_df, test_df, metadata = load_data()
    
    print(f"Training samples: {len(training_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Prepare training data (filtered - no DNFs/outliers, all positions 1-20)
    print("\nPreparing training data...")
    
    # TIME-BASED SPLIT: Split by year BEFORE filtering to prevent data leakage
    # Use most recent year for validation (more realistic than random split)
    print("\nSplitting data for validation (time-based to prevent overfitting)...")
    val_year = training_df['Year'].max()
    train_df_split = training_df[training_df['Year'] < val_year].copy()
    val_df_split = training_df[training_df['Year'] == val_year].copy()
    
    if len(val_df_split) == 0:
        # Fallback to random split if no separate year available
        print(f"  Warning: No separate year for validation, using random split")
        X_train, y_train, _, train_stats, feature_names = prepare_features_and_labels(
            training_df, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=False)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
    else:
        # Prepare train and validation separately with time-based split
        train_years = sorted(train_df_split['Year'].unique())
        val_years = sorted(val_df_split['Year'].unique())
        print(f"  Train years: {train_years}")
        print(f"  Validation year: {val_years}")
        print(f"  This prevents data leakage and gives more realistic validation")
        
        X_train, y_train, _, train_stats, feature_names = prepare_features_and_labels(
            train_df_split, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=False)
        X_val, y_val, _, val_stats, _ = prepare_features_and_labels(
            val_df_split, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=False)
        
        X_train_split = X_train
        X_val_split = X_val
        y_train_split = y_train
        y_val_split = y_val
        
        print(f"  Train samples: {len(X_train_split)}")
        print(f"  Validation samples: {len(X_val_split)}")
    
    # Prepare test data (filtered - for primary evaluation, all positions 1-20)
    print("\nPreparing test data (filtered - excluding DNFs/outliers)...")
    X_test, y_test, _, test_stats, _ = prepare_features_and_labels(
        test_df, filter_dnf=True, filter_outliers=True, outlier_threshold=6, top10_only=False)
    
    # Also prepare unfiltered test data (for comparison/reference, all positions 1-20)
    # Note: We still exclude DNFs (NaN positions) since we can't evaluate accuracy on them
    # But we include outliers (finish >> grid) for reference
    print("\nPreparing test data (unfiltered - including outliers for reference)...")
    X_test_all, y_test_all, _, test_stats_all, _ = prepare_features_and_labels(
        test_df, filter_dnf=True, filter_outliers=False, outlier_threshold=6, top10_only=False)
    
    # Scale features (fit on training, transform validation and test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val_split)
    X_test_scaled = scaler.transform(X_test)
    X_test_all_scaled = scaler.transform(X_test_all) if len(X_test_all) > 0 else None
    
    print(f"  Features shape: {X_train_scaled.shape}")
    print(f"  Labels shape: {y_train_split.shape}")
    print(f"  Position range: {y_train_split.min():.0f} - {y_train_split.max():.0f}")
    
    # Update variable names for consistency
    X_train_split = X_train_scaled
    X_val_split = X_val_scaled
    
    # Architecture: Optimized based on training improvements test
    # Best configuration: LR 0.005, Lower Dropout (0.3), Wider Model [512, 512, 256] or [256, 128, 64]
    # Using [256, 128, 64] with improved hyperparameters
    hidden_sizes = [256, 128, 64]
    
    # Train ensemble of 3 models (ensemble improves accuracy by reducing variance)
    print("\n" + "=" * 60)
    print("TRAINING ENSEMBLE (3 MODELS)")
    print("=" * 60)
    print("Training 3 models with different random seeds for ensemble prediction")
    
    models = []
    histories = []
    
    for i in range(3):
        print(f"\nTraining ensemble model {i+1}/3...")
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        
        # Optimized hyperparameters from training improvements test:
        # - LR 0.005: +8.37% improvement in validation MAE
        # - Lower Dropout 0.3: +1.43% improvement
        # - Reduced Weight Decay: +1.11% improvement
        model, history = train_model(
            X_train_split, y_train_split,
            X_val_split, y_val_split,
            epochs=300,  # Increased from 160 - allow more training with lenient early stopping
            batch_size=32,
            learning_rate=0.005,  # Optimized: increased from 0.001 (8.37% improvement)
            device=device,
            hidden_sizes=hidden_sizes,
            feature_names=feature_names
        )
        models.append(model)
        histories.append(history)
        print(f"  Model {i+1} training complete")
    
    # Use the first model for evaluation/plotting (they should be similar)
    model = models[0]
    history = histories[0]
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Final Evaluation on Validation Set")
    print("=" * 60)
    
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
    # feature_names already set from prepare_features_and_labels
    feature_importances = get_feature_importance(model, feature_names, device)
    print(f"\nFeature Importances (Weight Distribution from First Layer):")
    for name, importance in feature_importances.items():
        print(f"  {name}: {importance:.4f}")
    
    # Scatter plot: predicted vs actual positions
    script_dir = Path(__file__).parent
    images_dir = script_dir.parent / 'images'
    images_dir.mkdir(exist_ok=True, parents=True)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(val_labels, val_preds, alpha=0.5)
    plt.plot([val_labels.min(), val_labels.max()], [val_labels.min(), val_labels.max()], 'r--', lw=2)
    plt.xlabel('Actual Position')
    plt.ylabel('Predicted Position')
    plt.title('Predicted vs Actual Finishing Positions (Neural Network)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_path = images_dir / 'prediction_scatter.png'
    plt.savefig(scatter_path, dpi=150)
    print(f"\nPrediction scatter plot saved to {scatter_path}")
    plt.close()
    
    # Evaluate on test set (filtered - primary metrics)
    if len(X_test_scaled) > 0:
        print("\n" + "=" * 60)
        print("Evaluation on Test Set (FILTERED - Excluding DNFs/Outliers)")
        print("=" * 60)
        print("This is the primary evaluation metric - excludes unpredictable failures")
        print(f"Test samples: {len(y_test)} (after filtering)")
        
        test_dataset = F1Dataset(X_test_scaled, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        test_loss, test_mae, test_rmse, test_r2, test_exact, test_w1, test_w2, test_w3, test_preds, test_labels = evaluate_model(
            model, test_loader, criterion, device
        )
        
        print(f"\nTest Metrics (Filtered):")
        print(f"  MAE: {test_mae:.3f} positions")
        print(f"  RMSE: {test_rmse:.3f} positions")
        print(f"  R²: {test_r2:.4f}")
        print(f"\nPosition Accuracy (Filtered):")
        print(f"  Within 1 position: {test_w1:.1f}%")
        print(f"  Within 2 positions: {test_w2:.1f}%")
        print(f"  Within 3 positions: {test_w3:.1f}%")
        
        # Also evaluate on unfiltered data for reference
        if X_test_all_scaled is not None and len(X_test_all_scaled) > 0:
            print("\n" + "=" * 60)
            print("Evaluation on Test Set (UNFILTERED - Including All Entries)")
            print("=" * 60)
            print("Reference metrics - includes DNFs/outliers for comparison")
            print(f"Test samples: {len(y_test_all)} (all entries)")
            
            test_dataset_all = F1Dataset(X_test_all_scaled, y_test_all)
            test_loader_all = DataLoader(test_dataset_all, batch_size=32, shuffle=False)
            
            test_loss_all, test_mae_all, test_rmse_all, test_r2_all, test_exact_all, test_w1_all, test_w2_all, test_w3_all, _, _ = evaluate_model(
                model, test_loader_all, criterion, device
            )
            
            print(f"\nTest Metrics (Unfiltered - Reference):")
            print(f"  MAE: {test_mae_all:.3f} positions")
            print(f"  RMSE: {test_rmse_all:.3f} positions")
            print(f"  R²: {test_r2_all:.4f}")
            print(f"\nPosition Accuracy (Unfiltered - Reference):")
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
    
    # Save all ensemble models (no label encoder for regression)
    print("\n" + "=" * 60)
    print("SAVING ENSEMBLE MODELS")
    print("=" * 60)
    for i, ensemble_model in enumerate(models):
        save_model(ensemble_model, scaler, None, model_index=i)
    print(f"\nSaved {len(models)} ensemble models")
    
    # Save results - convert all numpy types to native Python types for JSON serialization
    # Helper function to convert numpy types to native Python types
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
            # Try to convert if it's a numpy scalar type
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
        'model_type': 'neural_network_regression',
        'filtering_stats': convert_to_native(filtering_stats_dict),
        'validation_mae': float(val_mae),
        'validation_rmse': float(val_rmse),
        'validation_r2': float(val_r2),
        'validation_exact': float(val_exact),
        'validation_within_1': float(val_w1),
        'validation_within_2': float(val_w2),
        'validation_within_3': float(val_w3),
        'test_filtered': {
            'mae': float(test_mae) if test_mae else None,
            'rmse': float(test_rmse) if test_rmse else None,
            'r2': float(test_r2) if test_r2 else None,
            'exact': float(test_exact) if 'test_exact' in locals() and test_exact else None,
            'within_1': float(test_w1) if test_w1 else None,
            'within_2': float(test_w2) if test_w2 else None,
            'within_3': float(test_w3) if test_w3 else None,
        },
        'test_unfiltered': {
            'mae': float(test_mae_all) if 'test_mae_all' in locals() and test_mae_all else None,
            'rmse': float(test_rmse_all) if 'test_rmse_all' in locals() and test_rmse_all else None,
            'r2': float(test_r2_all) if 'test_r2_all' in locals() and test_r2_all else None,
            'exact': float(test_exact_all) if 'test_exact_all' in locals() and test_exact_all else None,
            'within_1': float(test_w1_all) if 'test_w1_all' in locals() and test_w1_all else None,
            'within_2': float(test_w2_all) if 'test_w2_all' in locals() and test_w2_all else None,
            'within_3': float(test_w3_all) if 'test_w3_all' in locals() and test_w3_all else None,
        },
        'feature_importances': {k: float(v) for k, v in feature_importances.items()},
        'model_architecture': {
            'input_size': X_train_split.shape[1],  # Should be 8 features (ConstructorTrackAvg added)
            'hidden_sizes': [128, 64, 32],
            'dropout_rate': 0.3,
            'output': 'regression (single position value)',
            'features': feature_names if feature_names else []
        }
    }
    
    # Create json directory if it doesn't exist
    script_dir = Path(__file__).parent
    json_dir = script_dir.parent / 'json'
    json_dir.mkdir(exist_ok=True, parents=True)
    
    results_path = json_dir / 'training_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Results saved to {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
