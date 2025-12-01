"""
Visualize the F1 Neural Network architecture.
Supports multiple visualization methods:
1. torchsummary - Text summary (requires: pip install torchsummary)
2. torchviz - Graph visualization (requires: pip install torchviz graphviz)
3. Simple print - Built-in PyTorch model printing
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Import model architecture
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import importlib.util
spec = importlib.util.spec_from_file_location("train_module", parent_dir / "top20" / "train.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

F1NeuralNetwork = train_module.F1NeuralNetwork


def print_model_summary(model, input_size=(9,)):
    """Print a simple text summary of the model."""
    print("=" * 80)
    print("NEURAL NETWORK ARCHITECTURE SUMMARY")
    print("=" * 80)
    print(f"\nInput Size: {input_size[0]} features")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("\n" + "-" * 80)
    print("LAYER BREAKDOWN:")
    print("-" * 80)
    
    total_params = 0
    for i, (name, module) in enumerate(model.named_modules()):
        if len(list(module.children())) == 0:  # Leaf node
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > 0:
                total_params += num_params
                if isinstance(module, nn.Linear):
                    print(f"  {name}: Linear({module.in_features} -> {module.out_features})")
                    print(f"    Parameters: {num_params:,} (weights: {module.weight.numel():,}, bias: {module.bias.numel() if module.bias is not None else 0:,})")
                elif isinstance(module, nn.BatchNorm1d):
                    print(f"  {name}: BatchNorm1d({module.num_features})")
                    print(f"    Parameters: {num_params:,}")
                elif isinstance(module, nn.Dropout):
                    print(f"  {name}: Dropout(p={module.p})")
                    print(f"    Parameters: 0 (no learnable parameters)")
                elif isinstance(module, nn.ReLU):
                    print(f"  {name}: ReLU()")
                    print(f"    Parameters: 0 (no learnable parameters)")
    
    print("\n" + "-" * 80)
    print("ARCHITECTURE FLOW:")
    print("-" * 80)
    
    # Print the sequential structure
    print("\nModel Structure:")
    print(model)
    
    print("\n" + "=" * 80)


def torchsummary_visualization(model, input_size=(9,), device='cpu'):
    """Use torchsummary to print detailed model summary."""
    try:
        from torchsummary import summary
        print("=" * 80)
        print("TORCHSUMMARY VISUALIZATION")
        print("=" * 80)
        print("\nNote: Install with: pip install torchsummary")
        print("\nDetailed Model Summary:")
        print("-" * 80)
        # Convert torch.device to string if needed
        device_str = str(device) if isinstance(device, torch.device) else device
        summary(model, input_size, device=device_str)
        print("=" * 80)
        return True
    except ImportError:
        print("torchsummary not available. Install with: pip install torchsummary")
        return False
    except Exception as e:
        print(f"Error with torchsummary: {e}")
        return False


def torchviz_visualization(model, input_size=(9,), save_path=None, device='cpu'):
    """Use torchviz to create a graph visualization."""
    try:
        from torchviz import make_dot
        import graphviz
        
        print("=" * 80)
        print("TORCHVIZ GRAPH VISUALIZATION")
        print("=" * 80)
        print("\nNote: Requires: pip install torchviz graphviz")
        print("      Also need graphviz system package: https://graphviz.org/download/")
        
        # Create dummy input on the same device as model
        dummy_input = torch.randn(1, *input_size).to(device)
        
        # Forward pass to create computation graph
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        # Create visualization
        dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
        
        if save_path is None:
            save_path = Path(__file__).parent.parent / 'images' / 'model_architecture.png'
        
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Save as PNG
        dot.render(save_path.with_suffix(''), format='png', cleanup=True)
        print(f"\nGraph visualization saved to: {save_path}")
        print("=" * 80)
        return True
    except ImportError as e:
        print(f"torchviz not available: {e}")
        print("Install with: pip install torchviz")
        print("Also install graphviz system package: https://graphviz.org/download/")
        return False
    except Exception as e:
        print(f"Error creating graph visualization: {e}")
        return False


def create_architecture_diagram(model, save_path=None):
    """Create a simple text-based architecture diagram."""
    if save_path is None:
        save_path = Path(__file__).parent.parent / 'images' / 'model_architecture.txt'
    
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("F1 NEURAL NETWORK ARCHITECTURE\n")
        f.write("=" * 80 + "\n\n")
        
        # Get input size
        first_layer = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                first_layer = module
                break
        
        if first_layer:
            input_size = first_layer.in_features
            f.write(f"Input Layer: {input_size} features\n")
            f.write("\n")
            
            # Track layer sizes
            layer_sizes = [input_size]
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    layer_sizes.append(module.out_features)
            
            # Create diagram
            f.write("Architecture Flow:\n")
            f.write("-" * 80 + "\n")
            for i in range(len(layer_sizes) - 1):
                f.write(f"  [{layer_sizes[i]}] -> Linear -> ReLU -> Dropout -> BatchNorm -> [{layer_sizes[i+1]}]\n")
            f.write(f"  [{layer_sizes[-1]}] -> Linear -> [1] (Output)\n")
            f.write("\n")
            
            f.write("Layer Details:\n")
            f.write("-" * 80 + "\n")
            layer_num = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    layer_num += 1
                    f.write(f"Layer {layer_num}: {name}\n")
                    f.write(f"  Type: Linear\n")
                    f.write(f"  Input Features: {module.in_features}\n")
                    f.write(f"  Output Features: {module.out_features}\n")
                    f.write(f"  Parameters: {sum(p.numel() for p in module.parameters()):,}\n")
                    f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\nArchitecture diagram saved to: {save_path}")


def matplotlib_visualization(model, save_path=None):
    """Create a matplotlib-based visual diagram of the neural network architecture."""
    try:
        print("=" * 80)
        print("MATPLOTLIB ARCHITECTURE VISUALIZATION")
        print("=" * 80)
        
        # Extract layer information
        layers = []
        layer_num = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_num += 1
                layers.append({
                    'name': f'Layer {layer_num}',
                    'input_size': module.in_features,
                    'output_size': module.out_features,
                    'params': sum(p.numel() for p in module.parameters())
                })
        
        if not layers:
            print("No layers found in model")
            return False
        
        # Create figure - layout: Input (top) -> Layers -> Output (bottom)
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 10)
        total_height = len(layers) + 3  # Input + layers + output + spacing
        ax.set_ylim(-1, total_height)
        ax.axis('off')
        
        # Title
        ax.text(5, total_height - 0.5, 'F1 Neural Network Architecture', 
                ha='center', va='center', fontsize=20, fontweight='bold')
        
        # Input layer (at the top)
        input_y = total_height - 1.5
        input_box = FancyBboxPatch((1, input_y - 0.3), 8, 0.6,
                                   boxstyle="round,pad=0.1",
                                   edgecolor='black', facecolor='lightgreen',
                                   linewidth=2, zorder=2)
        ax.add_patch(input_box)
        ax.text(5, input_y, f"Input\n{layers[0]['input_size']} features", 
               ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)
        
        # Draw layers (top to bottom: Layer 1, Layer 2, Layer 3, Layer 4)
        y_positions = []
        for i, layer in enumerate(layers):
            # Position layers below input, with spacing
            y_pos = total_height - 2.5 - i * 1.2
            y_positions.append(y_pos)
            
            # Layer box
            box = FancyBboxPatch((1, y_pos - 0.3), 8, 0.6,
                                boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor='lightblue',
                                linewidth=2, zorder=2)
            ax.add_patch(box)
            
            # Layer text
            layer_text = f"{layer['name']}\n{layer['input_size']} → {layer['output_size']}\n{layer['params']:,} params"
            ax.text(5, y_pos, layer_text, ha='center', va='center',
                   fontsize=11, fontweight='bold', zorder=3)
            
            # Activation/Dropout/BatchNorm labels (below each hidden layer)
            if i < len(layers) - 1:
                ax.text(5, y_pos - 0.6, 'ReLU → Dropout(0.4) → BatchNorm',
                       ha='center', va='top', fontsize=9, style='italic', color='gray')
        
        # Arrow from input to first layer
        if y_positions:
            arrow = FancyArrowPatch((5, input_y - 0.4), (5, y_positions[0] + 0.3),
                                   arrowstyle='->', lw=2, color='black', zorder=1)
            ax.add_patch(arrow)
        
        # Draw arrows between layers (after all positions are calculated)
        for i in range(len(layers) - 1):
            arrow = FancyArrowPatch((5, y_positions[i] - 0.4), (5, y_positions[i+1] + 0.3),
                                   arrowstyle='->', lw=2, color='black', zorder=1)
            ax.add_patch(arrow)
        
        # Output layer (at the bottom)
        output_y = y_positions[-1] - 1.2 if y_positions else 0.5
        output_box = FancyBboxPatch((1, output_y - 0.3), 8, 0.6,
                                    boxstyle="round,pad=0.1",
                                    edgecolor='black', facecolor='lightcoral',
                                    linewidth=2, zorder=2)
        ax.add_patch(output_box)
        ax.text(5, output_y, "Output\nPredicted Position (1-10)", 
               ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)
        
        # Arrow from last layer to output
        if y_positions:
            arrow = FancyArrowPatch((5, y_positions[-1] - 0.4), (5, output_y + 0.3),
                                   arrowstyle='->', lw=2, color='black', zorder=1)
            ax.add_patch(arrow)
        
        # Summary text (at the very bottom)
        total_params = sum(l['params'] for l in layers)
        summary_text = f"Total Parameters: {total_params:,}\nModel Size: ~{total_params * 4 / 1024:.2f} KB"
        ax.text(5, output_y - 1.0, summary_text, ha='center', va='top',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path is None:
            save_path = Path(__file__).parent.parent / 'images' / 'model_architecture.png'
        
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nArchitecture visualization saved to: {save_path}")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"Error creating matplotlib visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to visualize the model."""
    print("\n" + "=" * 80)
    print("F1 NEURAL NETWORK VISUALIZATION")
    print("=" * 80)
    
    # Model configuration (matching top10/train.py)
    input_size = 9  # Number of features
    hidden_sizes = [192, 96, 48]
    dropout_rate = 0.4
    
    # Create model
    print("\nCreating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = F1NeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        equal_init=False  # Using He initialization
    ).to(device)
    
    print(f"Model created on device: {device}")
    
    # Method 1: Simple print
    print("\n" + "=" * 80)
    print("METHOD 1: SIMPLE MODEL SUMMARY")
    print("=" * 80)
    print_model_summary(model, input_size=(input_size,))
    
    # Method 2: torchsummary (if available)
    print("\n")
    torchsummary_visualization(model, input_size=(input_size,), device=device)
    
    # Method 3: torchviz graph (if available)
    print("\n")
    torchviz_visualization(model, input_size=(input_size,), device=device)
    
    # Method 4: Matplotlib visualization (always works, no external dependencies)
    print("\n")
    matplotlib_visualization(model)
    
    # Method 5: Create text diagram
    print("\n" + "=" * 80)
    print("METHOD 5: TEXT-BASED ARCHITECTURE DIAGRAM")
    print("=" * 80)
    create_architecture_diagram(model)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print("\nTo install additional visualization tools:")
    print("  pip install torchsummary    # For detailed text summary")
    print("  pip install torchviz         # For graph visualization")
    print("  # Also install graphviz system package for torchviz:")
    print("  # Windows: choco install graphviz")
    print("  # Mac: brew install graphviz")
    print("  # Linux: sudo apt-get install graphviz")
    print("=" * 80)


if __name__ == "__main__":
    main()

