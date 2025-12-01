"""
Evaluation and metrics functions for F1 predictions.
"""

import numpy as np


def get_status(error: int) -> str:
    """Get status string based on error."""
    if error == 0:
        return "Exact"
    elif error == 1:
        return "Close"
    elif error == 2:
        return "Good"
    elif error <= 3:
        return "Fair"
    else:
        return "Poor"


def calculate_filtered_accuracy(predicted_scores: list, actual: list, grid_positions: list, 
                                outlier_threshold: int = 6) -> dict:
    """
    Calculate accuracy metrics with and without outliers (large position drops).
    
    Filters out cases where actual finish > grid + threshold (same as training filter).
    This shows how accuracy would be if we excluded unpredictable large drops.
    After filtering, re-ranks the remaining predictions and compares to actual positions.
    
    Args:
        predicted_scores: List of raw predicted scores (lower = better, will be ranked)
        actual: List of actual positions
        grid_positions: List of grid positions (AvgGridPosition)
        outlier_threshold: Threshold for large position drops (default: 6)
    
    Returns:
        Dictionary with 'full' and 'filtered' accuracy metrics
    """
    predicted_scores = np.array(predicted_scores)
    actual = np.array(actual)
    grid_positions = np.array(grid_positions)
    
    # Calculate position drops (actual - grid, positive = finished worse than started)
    position_drops = actual - grid_positions
    
    # Filter mask: exclude cases where finish > grid + threshold
    valid_mask = position_drops <= outlier_threshold
    
    # Full accuracy: rank all predictions and compare to actual
    # Lower predicted score = better rank (rank 1 is best)
    predicted_ranks_full = np.argsort(np.argsort(predicted_scores)) + 1
    errors_full = np.abs(predicted_ranks_full - actual)
    mae_full = np.mean(errors_full)
    exact_full = np.mean(predicted_ranks_full == actual) * 100
    within_1_full = np.mean(errors_full <= 1) * 100
    within_2_full = np.mean(errors_full <= 2) * 100
    within_3_full = np.mean(errors_full <= 3) * 100
    
    # Filtered accuracy (excluding outliers)
    if valid_mask.sum() > 0:
        # Get filtered predictions and actuals
        predicted_scores_filtered = predicted_scores[valid_mask]
        actual_filtered = actual[valid_mask]
        
        # Re-rank the filtered predictions (assign ranks 1, 2, 3, ... based on predicted scores)
        # Lower predicted score = better rank (rank 1 is best)
        predicted_ranks_filtered = np.argsort(np.argsort(predicted_scores_filtered)) + 1
        
        # Compare re-ranked predictions to actual positions
        errors_filtered = np.abs(predicted_ranks_filtered - actual_filtered)
        mae_filtered = np.mean(errors_filtered)
        exact_filtered = np.mean(predicted_ranks_filtered == actual_filtered) * 100
        within_1_filtered = np.mean(errors_filtered <= 1) * 100
        within_2_filtered = np.mean(errors_filtered <= 2) * 100
        within_3_filtered = np.mean(errors_filtered <= 3) * 100
        outliers_removed = (~valid_mask).sum()
    else:
        mae_filtered = mae_full
        exact_filtered = exact_full
        within_1_filtered = within_1_full
        within_2_filtered = within_2_full
        within_3_filtered = within_3_full
        outliers_removed = 0
    
    return {
        'full': {
            'mae': mae_full,
            'exact': exact_full,
            'within_1': within_1_full,
            'within_2': within_2_full,
            'within_3': within_3_full,
            'count': len(predicted_scores)
        },
        'filtered': {
            'mae': mae_filtered,
            'exact': exact_filtered,
            'within_1': within_1_filtered,
            'within_2': within_2_filtered,
            'within_3': within_3_filtered,
            'count': valid_mask.sum(),
            'outliers_removed': outliers_removed
        }
    }

