"""
Model loading and prediction functions for F1 predictions.
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from top10.config import FEATURE_COLS


def handle_nan_values(X: np.ndarray) -> np.ndarray:
    """
    Handle NaN values in feature matrix.
    For GridPosition (AvgGridPosition) specifically, we use per-driver historical averages,
    so if all values are NaN, we use a reasonable default (10.5 = mid-field)
    instead of filling with the same mean for all drivers.
    """
    # Suppress warnings for empty slices
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        with np.errstate(all='ignore'):
            for i in range(X.shape[1]):
                # Check if all values in this column are NaN
                nan_mask = np.isnan(X[:, i])
                if nan_mask.all():
                    # All NaN - use a reasonable default (10.5 for grid position, mid-field)
                    # For other features, use 0 as default
                    X[:, i] = 10.5 if i == 4 else 0  # AvgGridPosition (GridPosition) is typically index 4
                else:
                    # Some NaN - fill with mean of non-NaN values
                    non_nan_values = X[~nan_mask, i]
                    if len(non_nan_values) > 0:
                        fill_val = np.nanmean(non_nan_values)
                        if np.isnan(fill_val) or np.isinf(fill_val):
                            # Fallback to median if mean is NaN
                            fill_val = np.nanmedian(non_nan_values)
                            if np.isnan(fill_val) or np.isinf(fill_val):
                                fill_val = 0
                        X[:, i] = np.nan_to_num(X[:, i], nan=fill_val)
                    else:
                        # Shouldn't happen, but just in case
                        X[:, i] = 0
    return X


def make_predictions(X_scaled: np.ndarray, model, model_type: str, device=None) -> np.ndarray:
    """
    Make predictions using model(s).
    If model is a list (ensemble), averages predictions from all models.
    """
    if model_type == 'neural_network':
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            
            # Check if model is a list (ensemble)
            if isinstance(model, list):
                # Ensemble: average predictions from all models
                all_predictions = []
                for m in model:
                    pred = m(X_tensor).cpu().numpy()
                    if pred.ndim > 1:
                        pred = pred.flatten()
                    all_predictions.append(pred)
                
                # Average predictions
                predictions = np.mean(all_predictions, axis=0)
            else:
                # Single model
                predictions = model(X_tensor).cpu().numpy()
                if predictions.ndim > 1:
                    predictions = predictions.flatten()
            
            return predictions
    else:
        return model.predict(X_scaled)


class F1NeuralNetwork(nn.Module):
    """Neural Network model definition (must match training - regression)."""
    
    def __init__(self, input_size=9, hidden_sizes=[64, 32], dropout_rate=0.4, equal_init=False):
        super(F1NeuralNetwork, self).__init__()

        layers = []
        prev_size = input_size

        # Must match top10/train.py: Linear -> BatchNorm -> ReLU -> Dropout
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.network(x)
        # Squeeze only the last dimension (output dimension), not batch dimension
        if output.dim() > 1:
            return output.squeeze(-1)  # Squeeze last dimension only
        return output.squeeze()


def load_delta_ensemble(model_dir, scaler_name, meta_name, fallback_model_name=None):
    """
    Shared loader for delta-model ensembles (post-quali and pre-quali).
    Returns (model_or_models, scaler, meta) or (None, None, None).
    """
    import json as _json

    if model_dir is None:
        script_dir = Path(__file__).parent
        model_dir = script_dir.parent / 'models'
    else:
        model_dir = Path(model_dir)

    scaler_path = model_dir / scaler_name
    meta_path = model_dir / meta_name
    model_path = model_dir / fallback_model_name if fallback_model_name else None
    if not scaler_path.exists() or not meta_path.exists():
        return None, None, None

    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = _json.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        device = torch.device('cpu')

        # Ensemble (preferred): meta lists the member files; deltas are averaged
        # at prediction time (make_predictions handles model lists)
        member_paths = [model_dir / f for f in meta.get('ensemble_files', [])]
        member_paths = [p for p in member_paths if p.exists()]
        if not member_paths and model_path is not None and model_path.exists():
            member_paths = [model_path]
        if not member_paths:
            return None, None, None

        n_feats = len(meta.get('features', []))
        models = []
        for p in member_paths:
            checkpoint = torch.load(p, map_location=device)
            input_size = checkpoint['network.0.weight'].shape[1]
            if input_size != n_feats:
                print(f"Warning: delta model {p.name} expects {input_size} features but meta "
                      f"lists {n_feats} - ignoring. Retrain with top10/train.py.")
                return None, None, None
            m = F1NeuralNetwork(input_size=input_size, hidden_sizes=[64, 32], dropout_rate=0.4).to(device)
            m.load_state_dict(checkpoint)
            m.eval()
            models.append(m)

        model = models if len(models) > 1 else models[0]
        print(f"Loaded delta {'ensemble of ' + str(len(models)) if len(models) > 1 else 'model'} "
              f"from {meta_name} (input_size={n_feats})")
        return model, scaler, meta
    except Exception as e:
        print(f"Warning: could not load delta model ({meta_name}: {e})")
        return None, None, None


def load_postquali_model(model_dir: str = None):
    """Post-quali delta ensemble: score = actual-grid rank + mean ensemble delta.
    Returns (model_or_models, scaler, meta) or (None, None, None)."""
    return load_delta_ensemble(model_dir, 'scaler_top10_postquali.pkl', 'postquali_meta.json',
                               fallback_model_name='f1_predictor_model_top10_postquali.pth')


def load_prequali_delta_model(model_dir: str = None):
    """Pre-quali delta ensemble: score = form-order (season-avg grid) rank +
    mean ensemble delta. Used for future races where no quali data exists.
    Returns (model_or_models, scaler, meta) or (None, None, None)."""
    return load_delta_ensemble(model_dir, 'scaler_top10_prequali.pkl', 'prequali_meta.json')


def load_model(model_dir: str = None, model_type: str = 'neural_network', auto_fallback: bool = True):
    """
    Load trained model(s) and scaler.
    If ensemble models exist, loads all 3 and returns them as a list.
    
    Args:
        model_dir: Directory containing model files (default: ../models relative to script)
        model_type: 'neural_network' or 'random_forest'
        auto_fallback: If True, try the other model type if requested one is not found
        
    Returns:
        Tuple of (model(s), scaler, model_type, device)
        If ensemble models exist, model(s) is a list of 3 models, otherwise a single model
    """
    if model_dir is None:
        # Resolve path relative to this script's location
        script_dir = Path(__file__).parent
        model_dir = script_dir.parent / 'models'
    else:
        model_dir = Path(model_dir)
    
    if model_type == 'neural_network':
        nn_scaler_path = model_dir / 'scaler_top10.pkl'
        
        # Check for single model first (preferred - new training saves single model)
        single_model_path = model_dir / 'f1_predictor_model_top10.pth'
        
        if single_model_path.exists() and nn_scaler_path.exists():
            # Load single model
            print("Loading model...")
            with open(nn_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            device = torch.device('cpu')
            
            # Load checkpoint to determine input size from first layer weight shape
            checkpoint = torch.load(single_model_path, map_location=device)
            # Check the shape of the first layer weight: [hidden_size, input_size]
            first_layer_weight_shape = checkpoint['network.0.weight'].shape
            input_size = first_layer_weight_shape[1]  # Second dimension is input size
            
            model = F1NeuralNetwork(
                input_size=input_size,
                hidden_sizes=[64, 32],
                dropout_rate=0.4
            ).to(device)
            model.load_state_dict(checkpoint)
            model.eval()
            
            print(f"Loaded model (input_size={input_size})")
            
            # Fail fast if model input size does not match configured feature count
            expected_features = len(FEATURE_COLS)
            if input_size != expected_features:
                raise ValueError(
                    f"\n{'='*70}\n"
                    f"ERROR: Model feature mismatch! Model expects {input_size} features, "
                    f"but code is configured for {expected_features} features.\n"
                    f"\nThis is an old model file. Please delete it and retrain:\n"
                    f"  1. Delete old model files in {model_dir}:\n"
                    f"     - f1_predictor_model_top10.pth\n"
                    f"     - f1_predictor_model_top10_ensemble_*.pth\n"
                    f"     - scaler_top10.pkl\n"
                    f"  2. Retrain with: python top10/train.py\n"
                    f"{'='*70}"
                )
            
            # Verify scaler expects the same number of features
            scaler_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None
            if scaler_features is not None and scaler_features != expected_features:
                raise ValueError(
                    f"\n{'='*70}\n"
                    f"ERROR: Scaler feature mismatch! Scaler expects {scaler_features} features, "
                    f"but code is configured for {expected_features} features.\n"
                    f"\nThis is an old scaler file. Please delete it and retrain:\n"
                    f"  1. Delete old scaler file: {nn_scaler_path}\n"
                    f"  2. Retrain with: python top10/train.py\n"
                    f"{'='*70}"
                )
            
            return model, scaler, 'neural_network', device
        
        # Fallback: Check for ensemble models (backward compatibility - but will fail if old)
        ensemble_models = []
        for i in range(3):
            ensemble_path = model_dir / f'f1_predictor_model_top10_ensemble_{i}.pth'
            if ensemble_path.exists():
                ensemble_models.append(ensemble_path)
        
        if len(ensemble_models) == 3 and nn_scaler_path.exists():
            # Load ensemble models
            print("Loading ensemble models (3 models)...")
            with open(nn_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            device = torch.device('cpu')
            models = []
            
            for i, model_path in enumerate(ensemble_models):
                # Load checkpoint to determine input size from first layer weight shape
                checkpoint = torch.load(model_path, map_location=device)
                # Check the shape of the first layer weight: [hidden_size, input_size]
                first_layer_weight_shape = checkpoint['network.0.weight'].shape
                input_size = first_layer_weight_shape[1]  # Second dimension is input size
                
                model = F1NeuralNetwork(
                    input_size=input_size,
                    hidden_sizes=[64, 32],
                    dropout_rate=0.4
                ).to(device)
                model.load_state_dict(checkpoint)
                model.eval()
                models.append(model)
                print(f"  Loaded ensemble model {i+1}/3 (input_size={input_size})")
            
            print("Ensemble models loaded successfully")
            
            # Fail fast if old models detected (expect 11 features instead of 9)
            for i, m in enumerate(models):
                model_input_size = m.network[0].weight.shape[1]
                if model_input_size != 9:
                    raise ValueError(
                        f"\n{'='*70}\n"
                        f"ERROR: Old ensemble model {i+1} detected! Model expects {model_input_size} features, but code uses 9 features.\n"
                        f"\nThese are old model files. Please delete them and retrain:\n"
                        f"  1. Delete old model files in {model_dir}:\n"
                        f"     - f1_predictor_model_top10_ensemble_*.pth\n"
                        f"     - scaler_top10.pkl\n"
                        f"  2. Retrain with: python top10/train.py\n"
                        f"{'='*70}"
                    )
            
            # Verify scaler expects 9 features
            scaler_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None
            if scaler_features is not None and scaler_features != 9:
                raise ValueError(
                    f"\n{'='*70}\n"
                    f"ERROR: Old scaler detected! Scaler expects {scaler_features} features, but code uses 9 features.\n"
                    f"\nThis is an old scaler file. Please delete it and retrain:\n"
                    f"  1. Delete old scaler file: {nn_scaler_path}\n"
                    f"  2. Retrain with: python top10/train.py\n"
                    f"{'='*70}"
                )
            
            return models, scaler, 'neural_network', device
        elif auto_fallback:
            # Try Random Forest as fallback
            print(f"Warning: Neural network model not found")
            print("Attempting to load Random Forest model instead...")
            model_type = 'random_forest'  # Switch to try RF
        else:
            raise FileNotFoundError(f"Model not found. Run top10/train.py first.")
    
    # Try Random Forest (either requested or as fallback)
    if model_type == 'random_forest':
        rf_model_path = model_dir / 'f1_predictor_model_rf.pkl'
        rf_scaler_path = model_dir / 'scaler_rf.pkl'
        
        if rf_model_path.exists() and rf_scaler_path.exists():
            with open(rf_model_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(rf_scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            return model, scaler, 'random_forest', None
        elif auto_fallback:
            raise FileNotFoundError(
                f"Neither model found!\n"
                f"  Neural Network: {model_dir / 'f1_predictor_model_top10.pth'} (not found)\n"
                f"  Random Forest: {rf_model_path} (not found)\n"
                f"Please train at least one model:\n"
                f"  python top10/train.py (for neural network)\n"
                f"  python train_rf.py (for random forest)"
            )
        else:
            raise FileNotFoundError(f"Model not found at {rf_model_path}. Run train_rf.py first.")
    
    raise ValueError(f"Unknown model type: {model_type}")

