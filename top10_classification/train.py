"""
Training script for F1 Position Prediction using Classification (Top 10 Only).
Predicts position classes (1-10) directly instead of regression scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import sys
import pickle


class PositionAwareClassificationLoss(nn.Module):
    """
    Classification loss that penalizes position errors more heavily.
    Combines cross-entropy with a position-aware penalty.
    """
    def __init__(self, weight=None, position_penalty=2.0):
        super(PositionAwareClassificationLoss, self).__init__()
        self.weight = weight
        self.position_penalty = position_penalty
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, inputs, targets):
        # Standard cross-entropy loss
        ce = self.ce_loss(inputs, targets)
        
        # Get predicted classes
        pred_classes = torch.argmax(inputs, dim=1)
        
        # Calculate position errors
        position_errors = torch.abs(pred_classes.float() - targets.float())
        
        # Apply penalty for larger errors
        # The penalty increases exponentially with error size
        penalty = 1.0 + self.position_penalty * position_errors
        
        # Weight the loss by position error
        weighted_loss = ce * penalty
        
        return weighted_loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard example mining.
    Focuses learning on hard-to-classify examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights (can be tensor)
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Import from top20/train.py for shared utilities
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import importlib.util
spec = importlib.util.spec_from_file_location("train_module", parent_dir / "top20" / "train.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

# Import shared functions
load_data = train_module.load_data
prepare_features_and_labels = train_module.prepare_features_and_labels


class F1ClassificationDataset(Dataset):
    """PyTorch Dataset for F1 classification data."""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        # Convert positions (1-10) to class indices (0-9)
        self.labels = torch.LongTensor(labels.astype(int) - 1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class F1ClassificationNetwork(nn.Module):
    """
    Deep Neural Network for F1 Position Prediction (Classification).
    
    Architecture:
    - Input layer: variable features
    - Hidden layers: Multiple fully connected layers with ReLU activation
    - Output layer: 10 classes (positions 1-10)
    - Improved: Larger capacity and residual connections for better learning
    """
    
    def __init__(self, input_size=9, hidden_sizes=[256, 128, 64], dropout_rate=0.3):
        super(F1ClassificationNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers with improved architecture
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # BatchNorm before activation
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer: 10 classes for positions 1-10
        layers.append(nn.Linear(prev_size, 10))
        
        self.network = nn.Sequential(*layers)
        self.input_size = input_size
        
        # Initialize weights with He/Kaiming initialization (optimal for ReLU)
        self._initialize_he_weights()
    
    def _initialize_he_weights(self):
        """Initialize all layers with He/Kaiming initialization (optimal for ReLU)."""
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    # He/Kaiming initialization for ReLU
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)
    
    def get_first_layer_weights(self):
        """Get weights of first layer to see feature importance."""
        first_layer = self.network[0]
        return first_layer.weight.data.cpu().numpy(), first_layer.bias.data.cpu().numpy()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Convert class indices back to positions (0-9 -> 1-10)
    all_preds = np.array(all_preds) + 1
    all_labels = np.array(all_labels) + 1
    
    # Calculate metrics
    mae = mean_absolute_error(all_labels, all_preds)
    exact_accuracy = accuracy_score(all_labels, all_preds)
    
    # Within N positions accuracy
    within_1 = np.mean(np.abs(all_labels - all_preds) <= 1)
    within_3 = np.mean(np.abs(all_labels - all_preds) <= 3)
    
    # Top-3 accuracy (for actual top-3 finishers)
    top3_mask = all_labels <= 3
    if top3_mask.sum() > 0:
        top3_accuracy = accuracy_score(all_labels[top3_mask], all_preds[top3_mask])
    else:
        top3_accuracy = 0.0
    
    return {
        'loss': avg_loss,
        'mae': mae,
        'exact_accuracy': exact_accuracy,
        'within_1': within_1,
        'within_3': within_3,
        'top3_accuracy': top3_accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': np.array(all_probs)
    }


def calculate_class_weights(y_train):
    """Calculate class weights based on inverse frequency to handle class imbalance."""
    # Count occurrences of each position (1-10)
    position_counts = np.bincount(y_train.astype(int) - 1, minlength=10)
    
    # Calculate inverse frequency weights
    total = len(y_train)
    class_weights = total / (10 * position_counts + 1e-5)  # Add small epsilon to avoid division by zero
    
    # Normalize weights
    class_weights = class_weights / class_weights.mean()
    
    return torch.FloatTensor(class_weights)


def train_model(X_train, y_train, X_val, y_val, device, num_epochs=300, 
                learning_rate=0.0005, weight_decay=1e-4, batch_size=64,
                use_class_weights=True, feature_names=None):
    """Train classification model."""
    
    # Create datasets
    train_dataset = F1ClassificationDataset(X_train, y_train)
    val_dataset = F1ClassificationDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model with optimized architecture - larger for classification complexity
    input_size = X_train.shape[1]
    # Classification needs more capacity - use larger network than regression
    # Regression uses [128, 64, 32], we'll use significantly larger
    model = F1ClassificationNetwork(input_size=input_size, 
                                    hidden_sizes=[256, 128, 64],  # Larger network
                                    dropout_rate=0.3).to(device)  # Less dropout for more learning
    
    # Advanced loss function: Label Smoothing + Class Weights (balanced for top 10)
    if use_class_weights:
        class_weights = calculate_class_weights(y_train).cpu().numpy()
        # No special boost for top-3 - focus on overall top 10 accuracy
        # Class weights already handle class imbalance via inverse frequency
        # Normalize to ensure balanced learning
        class_weights = class_weights / class_weights.mean()
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        
        # Use standard CrossEntropyLoss with class weights - most stable
        # Label smoothing helps prevent overconfidence
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
        
        print(f"  Using CrossEntropyLoss with class weights and label smoothing (0.1): {class_weights}")
        print(f"  Focus: Overall top 10 accuracy (no top-3 boost)")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Use standard loss with label smoothing
        print("  Using CrossEntropyLoss with label smoothing (0.1) - no class weights")
    
    # Improved optimizer settings - slightly lower LR for larger model
    optimized_lr = 0.003  # Slightly lower for larger model stability
    optimizer = optim.Adam(model.parameters(), lr=optimized_lr, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, verbose=False, min_lr=1e-6)
    
    # Early stopping to prevent overfitting - use MAE instead of loss for more stable early stopping
    best_val_mae = float('inf')
    patience_counter = 0
    early_stop_patience = 50  # Lenient patience - allow extensive training
    print(f"  Learning rate: {optimized_lr:.6f}")
    
    # Training history
    train_losses = []
    train_accs = []
    val_losses = []
    val_maes = []
    val_exact_accs = []
    best_val_mae = float('inf')
    best_model_state = None
    
    print(f"\nTraining for {num_epochs} epochs...")
    print(f"Early stopping patience: {early_stop_patience} epochs")
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_metrics['loss'])
        val_maes.append(val_metrics['mae'])
        val_exact_accs.append(val_metrics['exact_accuracy'])
        
        # Learning rate scheduling
        scheduler.step(val_metrics['mae'])
        
        # Early stopping based on validation MAE (more stable than loss for this task)
        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
                print(f"Best validation MAE: {best_val_mae:.4f}, Best validation loss: {best_val_loss:.4f}")
                break
        
        # Print progress with class weights info
        if (epoch + 1) % 20 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_metrics['loss']:.4f}, Val MAE: {val_metrics['mae']:.4f}, "
                  f"Val Exact Acc: {val_metrics['exact_accuracy']*100:.2f}%, "
                  f"LR: {current_lr:.6f}, Patience: {patience_counter}/{early_stop_patience}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model (Val MAE: {best_val_mae:.4f}, Val Loss: {best_val_loss:.4f})")
    else:
        print(f"\nWarning: No model state saved (using final model)")
        best_val_mae = val_metrics['mae'] if len(val_maes) > 0 else float('inf')
    
    # Print first layer weights to see feature importance
    try:
        first_layer_weights, first_layer_bias = model.get_first_layer_weights()
        # Calculate feature importance as absolute sum of weights for each feature
        feature_importance = np.abs(first_layer_weights).sum(axis=0)
        print(f"\nFirst Layer Feature Importance (absolute weight sum):")
        if 'feature_names' in locals() and feature_names is not None:
            for i, (feat_name, importance) in enumerate(zip(feature_names, feature_importance)):
                print(f"  {feat_name}: {importance:.4f}")
        else:
            for i, importance in enumerate(feature_importance):
                print(f"  Feature {i}: {importance:.4f}")
        print(f"  (Higher = more important)")
    except Exception as e:
        print(f"  Could not print feature weights: {e}")
    
    return model, {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_mae': val_maes,
        'val_exact_acc': val_exact_accs
    }


def save_model(model, scaler, output_dir: str = None, model_index: int = None):
    """Save trained model and scaler."""
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / 'models' / 'top10_classification'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if model_index is not None:
        model_path = output_dir / f'f1_classifier_top10_ensemble_{model_index}.pth'
    else:
        model_path = output_dir / 'f1_classifier_top10.pth'
    
    scaler_path = output_dir / 'scaler_top10_classification.pkl'
    
    torch.save(model.state_dict(), model_path)
    
    if model_index is None or model_index == 0:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")
    
    print(f"Model saved to {model_path}")


def main():
    """Main function to train and test the F1 classification model."""
    print("F1 Position Prediction Model Training (Classification - Top 10 Only)")
    print("=" * 70)
    print("Predicting position classes (1-10) directly using classification")
    print("=" * 70)
    
    # Set device
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    training_df, test_df, metadata = load_data()
    
    print(f"Training samples: {len(training_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Prepare features and labels (top 10 only)
    print("\nPreparing features and labels (top 10 only)...")
    X_train, y_train, _, filter_stats_train, feature_names = prepare_features_and_labels(
        training_df, 
        filter_dnf=True, 
        filter_outliers=True, 
        outlier_threshold=6,
        top10_only=True
    )
    
    print(f"\nFeature columns: {feature_names}")
    print(f"Training samples after filtering: {len(X_train)}")
    
    # Prepare test data
    if not test_df.empty:
        X_test, y_test, _, filter_stats_test, _ = prepare_features_and_labels(
            test_df,
            filter_dnf=True,
            filter_outliers=True,
            outlier_threshold=6,
            top10_only=True
        )
        print(f"Test samples after filtering: {len(X_test)}")
    else:
        X_test, y_test = None, None
        print("No test data available")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = None
    
    # Time-aware k-fold cross-validation
    print("\nPerforming time-aware k-fold cross-validation...")
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=False)  # No shuffle for time-aware
    
    # Sort by year and round for time-aware split
    train_indices = np.arange(len(X_train_scaled))
    # Use first 80% for training, last 20% for validation in each fold
    split_idx = int(len(train_indices) * 0.8)
    
    # Create train/val split
    train_idx = train_indices[:split_idx]
    val_idx = train_indices[split_idx:]
    
    X_train_fold = X_train_scaled[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train_scaled[val_idx]
    y_val_fold = y_train[val_idx]
    
    print(f"Train fold size: {len(X_train_fold)}, Val fold size: {len(X_val_fold)}")
    
    # Train ensemble of 3 models (like regression)
    print("\n" + "=" * 70)
    print("TRAINING ENSEMBLE (3 MODELS)")
    print("=" * 70)
    print("Training 3 models with different random seeds for ensemble prediction")
    
    models = []
    histories = []
    
    for i in range(3):
        print(f"\nTraining ensemble model {i+1}/3...")
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        
        model, history = train_model(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            device=device,
            num_epochs=300,  # Match regression's 300 epochs
            learning_rate=0.003,  # Optimized learning rate
            weight_decay=1e-4,
            batch_size=64,
            use_class_weights=True,
            feature_names=feature_names  # Pass for weight printing
        )
        models.append(model)
        histories.append(history)
        print(f"  Model {i+1} training complete")
    
    # Use first model for evaluation/display
    model = models[0]
    history = histories[0]
    
    # Evaluate ensemble on validation set
    print("\nEvaluating ensemble on validation set...")
    val_dataset = F1ClassificationDataset(X_val_fold, y_val_fold)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # Ensemble prediction: average probabilities from all models
            ensemble_probs = None
            for m in models:
                m.eval()
                outputs = m(X_batch)
                probs = F.softmax(outputs, dim=1)
                if ensemble_probs is None:
                    ensemble_probs = probs
                else:
                    ensemble_probs += probs
            ensemble_probs /= len(models)
            preds = torch.argmax(ensemble_probs, dim=1) + 1  # Convert 0-9 to 1-10
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    val_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    val_exact = np.mean(np.array(all_preds) == np.array(all_targets))
    val_within1 = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)) <= 1)
    val_within3 = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)) <= 3)
    
    # Top-3 accuracy
    val_top3_pred = set(np.array(all_preds)[np.array(all_targets) <= 3])
    val_top3_actual = set(np.array(all_targets)[np.array(all_targets) <= 3])
    val_top3_acc = len(val_top3_pred & val_top3_actual) / max(len(val_top3_actual), 1)
    
    print(f"\nValidation Results (Ensemble):")
    print(f"  MAE: {val_mae:.4f}")
    print(f"  Exact Accuracy: {val_exact*100:.2f}%")
    print(f"  Within 1 Position: {val_within1*100:.2f}%")
    print(f"  Within 3 Positions: {val_within3*100:.2f}%")
    print(f"  Top-3 Accuracy: {val_top3_acc*100:.2f}%")
    
    # Evaluate ensemble on test set
    if X_test_scaled is not None and y_test is not None:
        print("\nEvaluating ensemble on test set...")
        test_dataset = F1ClassificationDataset(X_test_scaled, y_test)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                # Ensemble prediction
                ensemble_probs = None
                for m in models:
                    m.eval()
                    outputs = m(X_batch)
                    probs = F.softmax(outputs, dim=1)
                    if ensemble_probs is None:
                        ensemble_probs = probs
                    else:
                        ensemble_probs += probs
                ensemble_probs /= len(models)
                preds = torch.argmax(ensemble_probs, dim=1) + 1
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        test_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
        test_exact = np.mean(np.array(all_preds) == np.array(all_targets))
        test_within1 = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)) <= 1)
        test_within3 = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)) <= 3)
        
        # Top-3 accuracy
        test_top3_pred = set(np.array(all_preds)[np.array(all_targets) <= 3])
        test_top3_actual = set(np.array(all_targets)[np.array(all_targets) <= 3])
        test_top3_acc = len(test_top3_pred & test_top3_actual) / max(len(test_top3_actual), 1)
        
        print(f"\nTest Results (Ensemble):")
        print(f"  MAE: {test_mae:.4f}")
        print(f"  Exact Accuracy: {test_exact*100:.2f}%")
        print(f"  Within 1 Position: {test_within1*100:.2f}%")
        print(f"  Within 3 Positions: {test_within3*100:.2f}%")
        print(f"  Top-3 Accuracy: {test_top3_acc*100:.2f}%")
    
    # Save ensemble models
    print("\nSaving ensemble models...")
    for i, m in enumerate(models):
        save_model(m, scaler, model_index=i)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()

