"""
Generate a scatter plot of predicted vs actual positions for all races in the test set.
This provides a comprehensive view of model performance across all test data.
"""

from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from top10.model_loader import load_model
from top10.train import (
    F1Dataset,
    evaluate_model,
    prepare_features_and_labels,
    load_data,
)


def main():
    print("=" * 70)
    print("Generating Test Set Scatter Plot")
    print("=" * 70)
    
    # Set device
    device = torch.device('cpu')
    
    # Load model and scaler
    print("\nLoading model and scaler...")
    model, scaler, model_type, device_from_loader = load_model()
    if isinstance(model, list):
        model = model[0]  # Use first model if ensemble
    model.eval()
    
    # Load test data
    print("Loading test data...")
    _, test_df, _ = load_data()
    
    if test_df.empty:
        print("No test data available!")
        return
    
    # Prepare test features (filtered, top10 only)
    print("Preparing test features (filtered, top-10 positions only)...")
    X_test, y_test, _, test_stats, _ = prepare_features_and_labels(
        test_df,
        filter_dnf=True,
        filter_outliers=True,
        outlier_threshold=6,
        top10_only=True,
    )
    
    if len(X_test) == 0:
        print("No test samples after filtering!")
        return
    
    print(f"Test samples: {len(X_test)}")
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions (raw scores)
    print("Making predictions...")
    test_dataset = F1Dataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    criterion = nn.MSELoss()
    _, _, _, _, _, _, _, _, all_preds_raw, all_labels = evaluate_model(
        model, test_loader, criterion, device
    )
    
    all_preds_raw = np.array(all_preds_raw)
    all_labels = np.array(all_labels)
    
    # Reconstruct the filtered dataframe to get race information
    # We need to apply the same filtering that prepare_features_and_labels did
    test_df_filtered = test_df.copy()
    test_df_filtered = test_df_filtered[test_df_filtered['ActualPosition'].notna()]
    test_df_filtered = test_df_filtered[test_df_filtered['ActualPosition'] <= 10]
    test_df_filtered = test_df_filtered[
        ~(test_df_filtered['ActualPosition'] > test_df_filtered['GridPosition'] + 6)
    ]
    
    # Renumber positions per race (same as prepare_features_and_labels does)
    if 'Year' in test_df_filtered.columns and 'RoundNumber' in test_df_filtered.columns:
        if 'EventName' in test_df_filtered.columns:
            race_groups = test_df_filtered.groupby(['Year', 'RoundNumber', 'EventName'])
        else:
            race_groups = test_df_filtered.groupby(['Year', 'RoundNumber'])
        
        test_df_filtered = test_df_filtered.copy()
        for (race_key, group) in race_groups:
            sorted_indices = group.sort_values('ActualPosition').index
            for new_pos, idx in enumerate(sorted_indices, start=1):
                test_df_filtered.at[idx, 'ActualPosition'] = new_pos
    
    # Create a results dataframe with predictions and race info
    # The order should match since both use the same filtering
    results_df = test_df_filtered.copy()
    results_df = results_df.reset_index(drop=True)  # Reset index to match array order
    
    # Ensure we have the same number of rows
    if len(results_df) != len(all_preds_raw):
        print(f"Warning: DataFrame length ({len(results_df)}) doesn't match predictions length ({len(all_preds_raw)})")
        # Take only the matching rows
        min_len = min(len(results_df), len(all_preds_raw))
        results_df = results_df.iloc[:min_len].copy()
        all_preds_raw = all_preds_raw[:min_len]
        all_labels = all_labels[:min_len]
    
    results_df['PredictedRaw'] = all_preds_raw
    results_df['ActualPosition'] = all_labels
    results_df['_array_index'] = range(len(results_df))
    
    # Rank predictions within each race
    print("Ranking predictions within each race...")
    all_preds_ranked = np.zeros(len(all_preds_raw))
    
    if 'Year' in results_df.columns and 'EventName' in results_df.columns and 'RoundNumber' in results_df.columns:
        # Group by race and rank within each race
        for (year, event, round_num), group in results_df.groupby(['Year', 'EventName', 'RoundNumber']):
            # Get array indices for this race group
            group_array_indices = group['_array_index'].values
            if len(group_array_indices) > 0:
                group_preds = all_preds_raw[group_array_indices]
                # Rank predictions within this race (lower score = rank 1, best position)
                group_ranks = np.argsort(np.argsort(group_preds)) + 1
                all_preds_ranked[group_array_indices] = group_ranks
    else:
        # Fallback: rank all predictions together if race info not available
        all_preds_ranked = np.argsort(np.argsort(all_preds_raw)) + 1
    
    all_preds = all_preds_ranked
    
    # Create scatter plot
    print("Generating scatter plot...")
    images_dir = Path('images')
    images_dir.mkdir(exist_ok=True, parents=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate average predicted position for each actual position (trend line)
    unique_positions = np.unique(all_labels)
    avg_predicted = []
    std_predicted = []
    positions_list = []
    
    for pos in sorted(unique_positions):
        mask = all_labels == pos
        if mask.sum() > 0:
            avg_pred = np.mean(all_preds[mask])
            std_pred = np.std(all_preds[mask])
            avg_predicted.append(avg_pred)
            std_predicted.append(std_pred)
            positions_list.append(pos)
    
    # Plot trend line with error bars (the red line the user likes)
    ax.plot(positions_list, avg_predicted, 'o-', color='red', markersize=10, 
           linewidth=3, label='Average Predicted Position', zorder=10)
    ax.errorbar(positions_list, avg_predicted, yerr=std_predicted, 
               fmt='none', color='red', alpha=0.5, capsize=5, capthick=2, zorder=9)
    
    # Perfect prediction line (y=x)
    ax.plot([0.5, 10.5], [0.5, 10.5], 'r--', lw=2.5, label='Perfect Prediction', zorder=8)
    
    # Calculate and display metrics
    mae = np.mean(np.abs(all_labels - all_preds))
    rmse = np.sqrt(np.mean((all_labels - all_preds) ** 2))
    
    # Add text box with key metrics
    textstr = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nN = {len(all_labels)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
             verticalalignment='top', bbox=props, zorder=11)
    
    ax.set_xlabel('Actual Position', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Position', fontsize=13, fontweight='bold')
    ax.set_title('Predicted vs Actual Finishing Positions\n(2025 Season Test Races, Top-10 Only)', 
                fontsize=15, fontweight='bold')
    
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, zorder=1)
    ax.set_xlim([0.5, 10.5])
    ax.set_ylim([0.5, 10.5])
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 11))
    plt.tight_layout()
    
    # Save plot
    scatter_path = images_dir / 'prediction_scatter_top10.png'
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    print(f"\nScatter plot saved to {scatter_path}")
    plt.close()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
