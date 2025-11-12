"""
Quick script to check the data and see why HistoricalTrackAvgPosition has 0 importance.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load data
data_path = Path('data') / 'training_data.csv'
if data_path.exists():
    df = pd.read_csv(data_path)
    
    print("Data Statistics:")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"\nHistoricalTrackAvgPosition statistics:")
    print(f"  NaN count: {df['HistoricalTrackAvgPosition'].isna().sum()}")
    print(f"  NaN percentage: {df['HistoricalTrackAvgPosition'].isna().sum() / len(df) * 100:.1f}%")
    print(f"  Non-NaN count: {df['HistoricalTrackAvgPosition'].notna().sum()}")
    print(f"\nValue statistics (excluding NaN):")
    valid_values = df['HistoricalTrackAvgPosition'].dropna()
    if len(valid_values) > 0:
        print(f"  Mean: {valid_values.mean():.2f}")
        print(f"  Median: {valid_values.median():.2f}")
        print(f"  Min: {valid_values.min():.2f}")
        print(f"  Max: {valid_values.max():.2f}")
        print(f"  Std: {valid_values.std():.2f}")
        print(f"\nUnique values: {valid_values.nunique()}")
    else:
        print("  No valid values!")
    
    print(f"\nAfter filling NaN with median:")
    median_val = df['HistoricalTrackAvgPosition'].median()
    if pd.isna(median_val):
        filled = df['HistoricalTrackAvgPosition'].fillna(0)
        print(f"  (All NaN, filled with 0)")
        print(f"  Unique values after fill: {filled.nunique()}")
        print(f"  Value counts:\n{filled.value_counts().head(10)}")
    else:
        filled = df['HistoricalTrackAvgPosition'].fillna(median_val)
        print(f"  Median: {median_val:.2f}")
        print(f"  Unique values after fill: {filled.nunique()}")
    
    print(f"\nSeasonPoints statistics:")
    print(df['SeasonPoints'].describe())
    
    print(f"\nSeasonAvgFinish statistics:")
    print(df['SeasonAvgFinish'].describe())
    
else:
    print(f"Data file not found at {data_path}")
    print("Run collect_data.py first!")

