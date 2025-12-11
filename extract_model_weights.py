"""
Extract and save the weights from the trained top10 model.
This allows inspection of the learned parameters.
"""

from pathlib import Path
import torch
import numpy as np
import json

from top10.model_loader import load_model


def extract_weights(model):
    """Extract all weights and biases from the model."""
    weights_dict = {}
    
    for name, param in model.named_parameters():
        # Convert to numpy for easier handling
        param_data = param.data.cpu().numpy()
        
        # Store as list (for JSON serialization) and shape info
        weights_dict[name] = {
            'shape': list(param_data.shape),
            'data': param_data.tolist(),
            'mean': float(np.mean(param_data)),
            'std': float(np.std(param_data)),
            'min': float(np.min(param_data)),
            'max': float(np.max(param_data))
        }
    
    return weights_dict


def main():
    print("=" * 70)
    print("Extracting Model Weights")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model, scaler, model_type, device = load_model()
    
    if isinstance(model, list):
        print("Found ensemble model, extracting weights from first model...")
        model = model[0]
    
    model.eval()
    
    # Extract weights
    print("Extracting weights...")
    weights_dict = extract_weights(model)
    
    # Print summary
    print("\nModel Architecture:")
    print("-" * 70)
    for name, info in weights_dict.items():
        print(f"{name:40s} Shape: {str(info['shape']):20s} "
              f"Mean: {info['mean']:8.4f}  Std: {info['std']:8.4f}")
    
    # Save weights to JSON
    output_dir = Path('models')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    weights_json_path = output_dir / 'model_weights_top10.json'
    with open(weights_json_path, 'w', encoding='utf-8') as f:
        json.dump(weights_dict, f, indent=2)
    print(f"\nWeights saved to {weights_json_path}")
    
    # Also save as numpy format (more compact, preserves precision)
    weights_npy_path = output_dir / 'model_weights_top10.npz'
    weights_npz = {}
    for name, param in model.named_parameters():
        weights_npz[name] = param.data.cpu().numpy()
    np.savez(weights_npy_path, **weights_npz)
    print(f"Weights saved to {weights_npy_path} (numpy format)")
    
    # Print first layer weights (feature importances)
    print("\n" + "=" * 70)
    print("First Layer Weights (Feature Importances)")
    print("=" * 70)
    
    # Get feature names
    from top10.config import FEATURE_COLS
    
    first_layer_name = None
    first_layer_weights = None
    for name, param in model.named_parameters():
        if 'network.0.weight' in name or (first_layer_name is None and 'weight' in name):
            first_layer_name = name
            first_layer_weights = param.data.cpu().numpy()
            break
    
    if first_layer_weights is not None:
        # Calculate feature importance as absolute mean weight per feature
        feature_importance = np.mean(np.abs(first_layer_weights), axis=0)
        
        # Normalize to sum to 1
        feature_importance = feature_importance / np.sum(feature_importance)
        
        print("\nFeature Importances (from first layer weights):")
        print("-" * 70)
        for feat_name, importance in zip(FEATURE_COLS, feature_importance):
            print(f"{feat_name:30s} {importance:.4f}")
        
        # Save feature importances
        feature_imp_dict = {feat: float(imp) for feat, imp in zip(FEATURE_COLS, feature_importance)}
        feature_imp_path = output_dir / 'feature_importances_top10.json'
        with open(feature_imp_path, 'w', encoding='utf-8') as f:
            json.dump(feature_imp_dict, f, indent=2)
        print(f"\nFeature importances saved to {feature_imp_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

