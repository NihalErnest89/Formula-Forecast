"""
Utility script to report evaluation metrics for the Top 10 F1 prediction model.

Run from the project root:
    python evaluate_top10_metrics.py

This script:
- Loads the trained neural network model and scaler
- Loads the prepared training/test data
- Recomputes test metrics (filtered and unfiltered):
    - MAE, RMSE, R^2
    - Exact, within 1, within 2, within 3 positions
- Reads cross-validation and validation metrics from json/training_results_top10.json
  and prints them in a paper-ready format.
"""

from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from top10.model_loader import load_model
from top10.train import (
    F1Dataset,
    evaluate_model,
    prepare_features_and_labels,
    load_data,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_results_json(root_dir: Path):
    results_path = root_dir / "json" / "training_results_top10.json"
    if not results_path.exists():
        print(f"Warning: results JSON not found at {results_path}")
        return None
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_cv_and_validation_from_json(results: dict) -> None:
    if results is None:
        return

    cv = results.get("k_fold_cv", {})
    avg = cv.get("average", {})
    final_val = results.get("final_model_validation", {})

    print_section("Cross-Validation (Time-Aware, 5 folds: 2020–2024)")
    print(f"MAE (avg):  {avg.get('mae', None):.3f} positions")
    print(f"RMSE (avg): {avg.get('rmse', None):.3f} positions")
    print(f"R² (avg):   {avg.get('r2', None):.4f}")
    print(f"Exact:      {avg.get('exact', None):.1f}%")
    print(f"Within 1:   {avg.get('within_1', None):.1f}%")
    print(f"Within 2:   {avg.get('within_2', None):.1f}%")
    print(f"Within 3:   {avg.get('within_3', None):.1f}%")

    print_section("Final Model – Validation Set (2024)")
    print(f"MAE:        {final_val.get('mae', None):.3f} positions")
    print(f"RMSE:       {final_val.get('rmse', None):.3f} positions")
    print(f"R²:         {final_val.get('r2', None):.4f}")
    print(f"Exact:      {final_val.get('exact', None):.1f}%")
    print(f"Within 1:   {final_val.get('within_1', None):.1f}%")
    print(f"Within 2:   {final_val.get('within_2', None):.1f}%")
    print(f"Within 3:   {final_val.get('within_3', None):.1f}%")


def evaluate_test_sets(model, scaler, device, test_df):
    # Filtered test set: excludes DNFs and large drops (finish > grid + 6), top 10 only
    X_test, y_test, _, test_stats, _ = prepare_features_and_labels(
        test_df,
        filter_dnf=True,
        filter_outliers=True,
        outlier_threshold=6,
        top10_only=True,
    )

    # Unfiltered test set: excludes DNFs, keeps outliers, top 10 only
    X_test_all, y_test_all, _, test_stats_all, _ = prepare_features_and_labels(
        test_df,
        filter_dnf=True,
        filter_outliers=False,
        outlier_threshold=6,
        top10_only=True,
    )

    # Scale features with the trained scaler
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else np.empty((0, 9))
    X_test_all_scaled = (
        scaler.transform(X_test_all) if len(X_test_all) > 0 else np.empty((0, 9))
    )

    criterion = nn.MSELoss()

    # Filtered metrics
    filtered_metrics = None
    if len(X_test_scaled) > 0:
        test_dataset = F1Dataset(X_test_scaled, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        (
            test_loss,
            test_mae,
            test_rmse,
            test_r2,
            test_exact,
            test_w1,
            test_w2,
            test_w3,
            _,
            _,
        ) = evaluate_model(model, test_loader, criterion, device)
        filtered_metrics = {
            "count": len(y_test),
            "mae": test_mae,
            "rmse": test_rmse,
            "r2": test_r2,
            "exact": test_exact,
            "within_1": test_w1,
            "within_2": test_w2,
            "within_3": test_w3,
            "filter_stats": test_stats,
        }

    # Unfiltered metrics
    unfiltered_metrics = None
    if len(X_test_all_scaled) > 0:
        test_dataset_all = F1Dataset(X_test_all_scaled, y_test_all)
        test_loader_all = DataLoader(test_dataset_all, batch_size=32, shuffle=False)
        (
            test_loss_all,
            test_mae_all,
            test_rmse_all,
            test_r2_all,
            test_exact_all,
            test_w1_all,
            test_w2_all,
            test_w3_all,
            _,
            _,
        ) = evaluate_model(model, test_loader_all, criterion, device)
        unfiltered_metrics = {
            "count": len(y_test_all),
            "mae": test_mae_all,
            "rmse": test_rmse_all,
            "r2": test_r2_all,
            "exact": test_exact_all,
            "within_1": test_w1_all,
            "within_2": test_w2_all,
            "within_3": test_w3_all,
            "filter_stats": test_stats_all,
        }

    return filtered_metrics, unfiltered_metrics


def print_test_metrics(filtered_metrics, unfiltered_metrics):
    print_section("Test Set – Filtered (Top 10, No DNFs, No Large Position Drops)")
    if filtered_metrics is None:
        print("No filtered test samples available.")
    else:
        print(f"Samples:    {filtered_metrics['count']}")
        print(f"MAE:        {filtered_metrics['mae']:.3f} positions")
        print(f"RMSE:       {filtered_metrics['rmse']:.3f} positions")
        print(f"R²:         {filtered_metrics['r2']:.4f}")
        print(f"Exact:      {filtered_metrics['exact']:.1f}%")
        print(f"Within 1:   {filtered_metrics['within_1']:.1f}%")
        print(f"Within 2:   {filtered_metrics['within_2']:.1f}%")
        print(f"Within 3:   {filtered_metrics['within_3']:.1f}%")

    print_section("Test Set – Unfiltered Reference (Top 10, DNFs Removed, Outliers Kept)")
    if unfiltered_metrics is None:
        print("No unfiltered test samples available.")
    else:
        print(f"Samples:    {unfiltered_metrics['count']}")
        print(f"MAE:        {unfiltered_metrics['mae']:.3f} positions")
        print(f"RMSE:       {unfiltered_metrics['rmse']:.3f} positions")
        print(f"R²:         {unfiltered_metrics['r2']:.4f}")
        print(f"Exact:      {unfiltered_metrics['exact']:.1f}%")
        print(f"Within 1:   {unfiltered_metrics['within_1']:.1f}%")
        print(f"Within 2:   {unfiltered_metrics['within_2']:.1f}%")
        print(f"Within 3:   {unfiltered_metrics['within_3']:.1f}%")

    if filtered_metrics is not None and unfiltered_metrics is not None:
        print_section("Filtered vs Unfiltered – Comparison (Top 10 Only)")
        print(
            f"MAE (filtered vs unfiltered): "
            f"{filtered_metrics['mae']:.3f} vs {unfiltered_metrics['mae']:.3f} "
            f"(improvement: {unfiltered_metrics['mae'] - filtered_metrics['mae']:.3f})"
        )
        print(
            f"Within 3 (filtered vs unfiltered): "
            f"{filtered_metrics['within_3']:.1f}% vs {unfiltered_metrics['within_3']:.1f}% "
            f"(improvement: {filtered_metrics['within_3'] - unfiltered_metrics['within_3']:.1f}%)"
        )


def main():
    root_dir = Path(__file__).parent

    # Load summary metrics from JSON (cross-validation, validation)
    results = load_results_json(root_dir)
    print_cv_and_validation_from_json(results)

    # Load trained model and scaler
    model, scaler, model_type, device = load_model()
    if model_type != "neural_network":
        raise ValueError(
            f"Expected neural_network model_type, got {model_type}. "
            f"Train the top-10 neural network model first."
        )

    # Load data and evaluate on test set
    training_df, test_df, metadata = load_data()

    filtered_metrics, unfiltered_metrics = evaluate_test_sets(
        model, scaler, device, test_df
    )
    print_test_metrics(filtered_metrics, unfiltered_metrics)


if __name__ == "__main__":
    main()


