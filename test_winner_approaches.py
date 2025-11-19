"""
Test different approaches to improve winner prediction accuracy.
Compares multiple strategies:
1. Weighted Loss Function (penalize winner prediction errors more)
2. Hybrid Approach (classification for winner, regression for others)
3. Winner-Specific Features (pole position, recent wins, etc.)
4. Ranking/Ordinal Approach (predict relative order)
5. Ensemble with Winner-Focused Model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import sys
import json

# Import from train.py
sys.path.insert(0, str(Path(__file__).parent))
from train import F1NeuralNetwork, prepare_features_and_labels, load_data, F1Dataset

device = torch.device('cpu')
print(f"Using device: {device}")

# --- Configuration ---
JSON_DIR = 'json'
Path(JSON_DIR).mkdir(exist_ok=True)

def convert_to_native(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native(item) for item in obj]
    elif hasattr(obj, 'item'):  # Other NumPy scalar types
        return obj.item()
    return obj


# --- Approach 1: Weighted Loss Function ---
class WeightedHuberLoss(nn.Module):
    """Huber loss with higher weight for position 1 (winner) errors."""
    def __init__(self, winner_weight=5.0, delta=1.0):
        super().__init__()
        self.winner_weight = winner_weight
        self.delta = delta
    
    def forward(self, pred, target):
        # Identify winners (position 1)
        is_winner = (target == 1.0).float()
        # Weight: winner_weight for winners, 1.0 for others
        weights = is_winner * (self.winner_weight - 1.0) + 1.0
        
        error = target - pred
        is_small = torch.abs(error) <= self.delta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * (torch.abs(error) - 0.5 * self.delta)
        loss = torch.where(is_small, squared_loss, linear_loss)
        
        return (loss * weights).mean()


# --- Approach 2: Hybrid Model (Classification for Winner) ---
class HybridF1Network(nn.Module):
    """Neural network with separate outputs: winner classification + position regression."""
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.4):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        # Winner classification head (binary: is winner or not)
        self.winner_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Probability of being winner
        )
        # Position regression head
        self.position_head = nn.Linear(prev_size, 1)
    
    def forward(self, x):
        shared = self.shared_layers(x)
        winner_prob = self.winner_head(shared)
        position = self.position_head(shared)
        return winner_prob, position


# --- Approach 3: Winner-Specific Features ---
def add_winner_features(df):
    """Add features that specifically help identify winners."""
    df = df.copy()
    
    # Feature 1: Is pole position (GridPosition == 1)
    df['IsPolePosition'] = (df['GridPosition'].fillna(10.0) == 1).astype(float)
    
    # Feature 2: Recent wins (count wins in last 5 races)
    # We'll approximate this using RecentForm - if RecentForm is very good, likely has recent wins
    # Better: calculate from actual race history if available
    df['RecentWinsApprox'] = ((df['RecentForm'].fillna(10.0) <= 2.0) & (df['RecentForm'].fillna(10.0) > 0)).astype(float)
    
    # Feature 3: Championship leader (SeasonPoints is highest)
    # We'll calculate this per race
    df['IsChampionshipLeader'] = 0.0
    for (year, event), group in df.groupby(['Year', 'EventName']):
        if 'SeasonPoints' in group.columns:
            max_points = group['SeasonPoints'].max()
            df.loc[group.index, 'IsChampionshipLeader'] = (group['SeasonPoints'] == max_points).astype(float)
    
    # Feature 4: Strong grid position (top 3)
    df['IsTop3Grid'] = (df['GridPosition'].fillna(10.0) <= 3).astype(float)
    
    # Feature 5: Dominant season performance (high points, low avg finish)
    # Handle NaN values in SeasonAvgFinish
    df['DominanceScore'] = (df['SeasonPoints'] / 100.0) * (1.0 / (df['SeasonAvgFinish'].fillna(10.0) + 1.0))
    # Normalize to 0-1 range (handle NaN and inf)
    df['DominanceScore'] = df['DominanceScore'].replace([np.inf, -np.inf], np.nan)
    if df['DominanceScore'].notna().any() and df['DominanceScore'].max() > 0:
        df['DominanceScore'] = df['DominanceScore'] / df['DominanceScore'].max()
    df['DominanceScore'] = df['DominanceScore'].fillna(0.0)
    
    # Fill any remaining NaN values with 0
    for col in ['IsPolePosition', 'RecentWinsApprox', 'IsChampionshipLeader', 'IsTop3Grid', 'DominanceScore']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    
    return df


# --- Approach 4: Ranking Loss (Ordinal Regression) ---
class RankingLoss(nn.Module):
    """Loss function that encourages correct ranking order."""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, pred, target):
        # For each pair of samples, ensure correct ordering
        # Winner (target=1) should have lower prediction than non-winners
        batch_size = pred.size(0)
        if batch_size < 2:
            # Fallback to MAE if batch too small
            return torch.mean(torch.abs(pred - target))
        
        # Pairwise ranking loss
        pred_expanded = pred.unsqueeze(1)  # [batch, 1]
        target_expanded = target.unsqueeze(1)  # [batch, 1]
        
        # Compare all pairs
        pred_diff = pred_expanded - pred_expanded.t()  # [batch, batch]
        target_diff = target_expanded - target_expanded.t()  # [batch, batch]
        
        # Loss when ranking is wrong
        wrong_ranking = (target_diff < 0) & (pred_diff > -self.margin)
        loss = torch.sum(torch.clamp(self.margin + pred_diff, min=0) * wrong_ranking.float())
        
        # Also include standard regression loss
        mae_loss = torch.mean(torch.abs(pred - target))
        
        return loss / (batch_size * (batch_size - 1)) + mae_loss


# --- Training Functions ---
def train_weighted_model(X_train, y_train, X_val, y_val, winner_weight=5.0, epochs=150):
    """Train model with weighted loss function."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    train_dataset = F1Dataset(X_train_scaled, y_train)
    val_dataset = F1Dataset(X_val_scaled, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = F1NeuralNetwork(input_size=X_train.shape[1], hidden_sizes=[256, 128, 64], dropout_rate=0.4)
    model.to(device)
    
    criterion = WeightedHuberLoss(winner_weight=winner_weight, delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_mae = float('inf')
    patience = 30
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(features).squeeze()
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                pred = model(features).squeeze()
                loss = criterion(pred, labels)
                val_loss += loss.item()
                val_preds.extend(pred.cpu().detach().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_mae = mean_absolute_error(val_labels, val_preds)
        scheduler.step(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return model, scaler, best_val_mae


def train_hybrid_model(X_train, y_train, X_val, y_val, epochs=150):
    """Train hybrid model (classification + regression)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create binary winner labels
    y_train_winner = (y_train == 1).astype(float)
    y_val_winner = (y_val == 1).astype(float)
    
    train_dataset = F1Dataset(X_train_scaled, y_train)
    val_dataset = F1Dataset(X_val_scaled, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = HybridF1Network(input_size=X_train.shape[1], hidden_sizes=[256, 128, 64], dropout_rate=0.4)
    model.to(device)
    
    winner_criterion = nn.BCELoss()
    position_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_mae = float('inf')
    patience = 30
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            labels_winner = (labels == 1.0).float()
            
            optimizer.zero_grad()
            winner_prob, position = model(features)
            winner_prob = winner_prob.squeeze()
            position = position.squeeze()
            
            winner_loss = winner_criterion(winner_prob, labels_winner)
            position_loss = position_criterion(position, labels)
            loss = winner_loss + position_loss  # Combined loss
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                labels_winner = (labels == 1.0).float()
                
                winner_prob, position = model(features)
                winner_prob = winner_prob.squeeze()
                position = position.squeeze()
                
                winner_loss = winner_criterion(winner_prob, labels_winner)
                position_loss = position_criterion(position, labels)
                loss = winner_loss + position_loss
                val_loss += loss.item()
                
                # Use position prediction, but if winner_prob > 0.5, force to 1
                pred = position.cpu().detach().numpy()
                winner_mask = winner_prob.cpu().detach().numpy() > 0.5
                pred[winner_mask] = 1.0
                val_preds.extend(pred)
                val_labels.extend(labels.cpu().numpy())
        
        val_mae = mean_absolute_error(val_labels, val_preds)
        scheduler.step(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return model, scaler, best_val_mae


def train_ranking_model(X_train, y_train, X_val, y_val, epochs=150):
    """Train model with ranking loss."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    train_dataset = F1Dataset(X_train_scaled, y_train)
    val_dataset = F1Dataset(X_val_scaled, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = F1NeuralNetwork(input_size=X_train.shape[1], hidden_sizes=[256, 128, 64], dropout_rate=0.4)
    model.to(device)
    
    criterion = RankingLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_mae = float('inf')
    patience = 30
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(features).squeeze()
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                pred = model(features).squeeze()
                loss = criterion(pred, labels)
                val_loss += loss.item()
                val_preds.extend(pred.cpu().detach().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_mae = mean_absolute_error(val_labels, val_preds)
        scheduler.step(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return model, scaler, best_val_mae


def evaluate_winner_prediction(model, scaler, X_test, y_test, is_hybrid=False):
    """Evaluate model on winner prediction specifically."""
    X_test_scaled = scaler.transform(X_test)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    test_dataset = F1Dataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            if is_hybrid:
                winner_prob, position = model(features)
                winner_prob = winner_prob.squeeze()
                position = position.squeeze()
                # Use position, but if winner_prob > 0.5, force to 1
                pred = position.cpu().detach().numpy()
                winner_mask = winner_prob.cpu().detach().numpy() > 0.5
                pred[winner_mask] = 1.0
            else:
                pred = model(features).squeeze().cpu().detach().numpy()
            
            # Clip to 1-10 range
            pred = np.clip(pred, 1, 10)
            all_preds.extend(pred)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Winner-specific metrics
    winner_mask = all_labels == 1
    if winner_mask.sum() > 0:
        winner_preds = all_preds[winner_mask]
        winner_labels = all_labels[winner_mask]
        
        winner_mae = mean_absolute_error(winner_labels, winner_preds)
        winner_rmse = np.sqrt(mean_squared_error(winner_labels, winner_preds))
        winner_within_1 = (np.abs(winner_preds - winner_labels) <= 1).sum() / len(winner_preds) * 100
        winner_within_2 = (np.abs(winner_preds - winner_labels) <= 2).sum() / len(winner_preds) * 100
        winner_exact = (np.abs(winner_preds - winner_labels) < 0.5).sum() / len(winner_preds) * 100
        winner_in_top3 = (winner_preds <= 3).sum() / len(winner_preds) * 100
        winner_in_top5 = (winner_preds <= 5).sum() / len(winner_preds) * 100
        
        return {
            'winner_mae': winner_mae,
            'winner_rmse': winner_rmse,
            'winner_within_1': winner_within_1,
            'winner_within_2': winner_within_2,
            'winner_exact': winner_exact,
            'winner_in_top3': winner_in_top3,
            'winner_in_top5': winner_in_top5,
            'num_winners': len(winner_preds)
        }
    else:
        return None


def main():
    print("F1 Winner Prediction - Alternative Approaches Comparison")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    training_df, test_df, _ = load_data(data_dir=str(data_dir))
    
    print(f"Training samples: {len(training_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Prepare training data (top 10 only)
    print("\nPreparing training data (top 10 positions only)...")
    X_train, y_train, _, train_stats, feature_names = prepare_features_and_labels(
        training_df, filter_dnf=True, filter_outliers=True, outlier_threshold=8, top10_only=True)
    
    X_test, y_test, _, test_stats, _ = prepare_features_and_labels(
        test_df, filter_dnf=True, filter_outliers=True, outlier_threshold=8, top10_only=True)
    
    print(f"Training on: {len(X_train)} samples (positions 1-10)")
    print(f"Testing on: {len(X_test)} samples (positions 1-10)")
    
    # Split training into train/val
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    results = {}
    
    # --- Approach 1: Weighted Loss (test different weights) ---
    print("\n" + "=" * 70)
    print("APPROACH 1: Weighted Loss Function")
    print("=" * 70)
    
    for winner_weight in [3.0, 5.0, 10.0]:
        print(f"\nTesting winner_weight = {winner_weight}...")
        model, scaler, val_mae = train_weighted_model(
            X_train_split, y_train_split, X_val_split, y_val_split,
            winner_weight=winner_weight, epochs=150
        )
        
        winner_metrics = evaluate_winner_prediction(model, scaler, X_test, y_test, is_hybrid=False)
        model.eval()
        with torch.no_grad():
            test_preds = model(torch.FloatTensor(scaler.transform(X_test)).to(device)).squeeze().cpu().detach().numpy()
        test_mae = mean_absolute_error(y_test, np.clip(test_preds, 1, 10))
        
        results[f'weighted_{winner_weight}'] = {
            'approach': 'Weighted Loss',
            'winner_weight': winner_weight,
            'val_mae': val_mae,
            'test_mae': test_mae,
            'winner_metrics': winner_metrics
        }
        print(f"  Val MAE: {val_mae:.3f}, Test MAE: {test_mae:.3f}")
        if winner_metrics:
            print(f"  Winner MAE: {winner_metrics['winner_mae']:.3f}, Winner Exact: {winner_metrics['winner_exact']:.1f}%")
    
    # --- Approach 2: Hybrid Model ---
    print("\n" + "=" * 70)
    print("APPROACH 2: Hybrid Model (Classification + Regression)")
    print("=" * 70)
    
    print("Training hybrid model...")
    model, scaler, val_mae = train_hybrid_model(
        X_train_split, y_train_split, X_val_split, y_val_split, epochs=150
    )
    
    winner_metrics = evaluate_winner_prediction(model, scaler, X_test, y_test, is_hybrid=True)
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.FloatTensor(scaler.transform(X_test)).to(device))[1].squeeze().cpu().detach().numpy()
    test_mae = mean_absolute_error(y_test, np.clip(test_preds, 1, 10))
    
    results['hybrid'] = {
        'approach': 'Hybrid (Classification + Regression)',
        'val_mae': val_mae,
        'test_mae': test_mae,
        'winner_metrics': winner_metrics
    }
    print(f"  Val MAE: {val_mae:.3f}, Test MAE: {test_mae:.3f}")
    if winner_metrics:
        print(f"  Winner MAE: {winner_metrics['winner_mae']:.3f}, Winner Exact: {winner_metrics['winner_exact']:.1f}%")
    
    # --- Approach 3: Winner-Specific Features ---
    print("\n" + "=" * 70)
    print("APPROACH 3: Winner-Specific Features")
    print("=" * 70)
    
    # Add winner features
    training_df_enhanced = add_winner_features(training_df)
    test_df_enhanced = add_winner_features(test_df)
    
    # Prepare features with new columns
    winner_feature_cols = feature_names + ['IsPolePosition', 'RecentWinsApprox', 'IsChampionshipLeader', 
                                          'IsTop3Grid', 'DominanceScore']
    
    # Extract features manually
    # First, ensure all feature columns exist and handle NaN
    for col in winner_feature_cols:
        if col not in training_df_enhanced.columns:
            training_df_enhanced[col] = 0.0
        if col not in test_df_enhanced.columns:
            test_df_enhanced[col] = 0.0
    
    # Fill NaN values in feature columns
    training_df_enhanced[winner_feature_cols] = training_df_enhanced[winner_feature_cols].fillna(0.0)
    test_df_enhanced[winner_feature_cols] = test_df_enhanced[winner_feature_cols].fillna(0.0)
    
    X_train_enhanced = training_df_enhanced[winner_feature_cols].values
    y_train_enhanced = training_df_enhanced['ActualPosition'].values
    
    # Filter top 10, DNF, outliers, and NaN in labels
    mask = (y_train_enhanced <= 10) & (~training_df_enhanced.get('IsDNF', pd.Series([False] * len(training_df_enhanced)))) & (~pd.isna(y_train_enhanced))
    X_train_enhanced = X_train_enhanced[mask]
    y_train_enhanced = y_train_enhanced[mask]
    
    # Remove any rows with NaN in features
    feature_mask = ~np.isnan(X_train_enhanced).any(axis=1)
    X_train_enhanced = X_train_enhanced[feature_mask]
    y_train_enhanced = y_train_enhanced[feature_mask]
    
    X_test_enhanced = test_df_enhanced[winner_feature_cols].values
    y_test_enhanced = test_df_enhanced['ActualPosition'].values
    mask = (y_test_enhanced <= 10) & (~test_df_enhanced.get('IsDNF', pd.Series([False] * len(test_df_enhanced)))) & (~pd.isna(y_test_enhanced))
    X_test_enhanced = X_test_enhanced[mask]
    y_test_enhanced = y_test_enhanced[mask]
    
    # Remove any rows with NaN in features
    feature_mask = ~np.isnan(X_test_enhanced).any(axis=1)
    X_test_enhanced = X_test_enhanced[feature_mask]
    y_test_enhanced = y_test_enhanced[feature_mask]
    
    X_train_split_enh, X_val_split_enh, y_train_split_enh, y_val_split_enh = train_test_split(
        X_train_enhanced, y_train_enhanced, test_size=0.2, random_state=42
    )
    
    print("Training model with winner-specific features...")
    model, scaler, val_mae = train_weighted_model(
        X_train_split_enh, y_train_split_enh, X_val_split_enh, y_val_split_enh,
        winner_weight=5.0, epochs=150
    )
    
    winner_metrics = evaluate_winner_prediction(model, scaler, X_test_enhanced, y_test_enhanced, is_hybrid=False)
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.FloatTensor(scaler.transform(X_test_enhanced)).to(device)).squeeze().cpu().detach().numpy()
    test_mae = mean_absolute_error(y_test_enhanced, np.clip(test_preds, 1, 10))
    
    results['winner_features'] = {
        'approach': 'Winner-Specific Features',
        'val_mae': val_mae,
        'test_mae': test_mae,
        'winner_metrics': winner_metrics,
        'features_added': ['IsPolePosition', 'RecentWinsApprox', 'IsChampionshipLeader', 'IsTop3Grid', 'DominanceScore']
    }
    print(f"  Val MAE: {val_mae:.3f}, Test MAE: {test_mae:.3f}")
    if winner_metrics:
        print(f"  Winner MAE: {winner_metrics['winner_mae']:.3f}, Winner Exact: {winner_metrics['winner_exact']:.1f}%")
    
    # --- Approach 4: Ranking Loss ---
    print("\n" + "=" * 70)
    print("APPROACH 4: Ranking Loss (Ordinal Regression)")
    print("=" * 70)
    
    print("Training model with ranking loss...")
    model, scaler, val_mae = train_ranking_model(
        X_train_split, y_train_split, X_val_split, y_val_split, epochs=150
    )
    
    winner_metrics = evaluate_winner_prediction(model, scaler, X_test, y_test, is_hybrid=False)
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.FloatTensor(scaler.transform(X_test)).to(device)).squeeze().cpu().detach().numpy()
    test_mae = mean_absolute_error(y_test, np.clip(test_preds, 1, 10))
    
    results['ranking'] = {
        'approach': 'Ranking Loss',
        'val_mae': val_mae,
        'test_mae': test_mae,
        'winner_metrics': winner_metrics
    }
    print(f"  Val MAE: {val_mae:.3f}, Test MAE: {test_mae:.3f}")
    if winner_metrics:
        print(f"  Winner MAE: {winner_metrics['winner_mae']:.3f}, Winner Exact: {winner_metrics['winner_exact']:.1f}%")
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    print("\nWinner Prediction Performance:")
    print(f"{'Approach':<40} {'Winner MAE':<12} {'Exact %':<10} {'Within 1 %':<12} {'Within 2 %':<12} {'Top 3 %':<10}")
    print("-" * 100)
    
    for name, result in results.items():
        if result['winner_metrics']:
            wm = result['winner_metrics']
            print(f"{result['approach']:<40} {wm['winner_mae']:<12.3f} {wm['winner_exact']:<10.1f} "
                  f"{wm['winner_within_1']:<12.1f} {wm['winner_within_2']:<12.1f} {wm['winner_in_top3']:<10.1f}")
    
    # Find best approach
    best_approach = None
    best_winner_mae = float('inf')
    for name, result in results.items():
        if result['winner_metrics'] and result['winner_metrics']['winner_mae'] < best_winner_mae:
            best_winner_mae = result['winner_metrics']['winner_mae']
            best_approach = (name, result)
    
    if best_approach:
        print(f"\n[+] Best approach for winner prediction: {best_approach[1]['approach']}")
        print(f"    Winner MAE: {best_winner_mae:.3f}")
        if best_approach[1]['winner_metrics']:
            print(f"    Winner Exact: {best_approach[1]['winner_metrics']['winner_exact']:.1f}%")
            print(f"    Winner Within 1: {best_approach[1]['winner_metrics']['winner_within_1']:.1f}%")
    
    # Save results
    results_path = Path(JSON_DIR) / 'winner_approaches_comparison.json'
    with open(results_path, 'w') as f:
        json.dump(convert_to_native(results), f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()

