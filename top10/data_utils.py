"""
Shared utilities for data loading, race list building, and prediction formatting.
Used by both api/app.py and generate_static_data.py to avoid duplicated logic.
"""

import pandas as pd
from pathlib import Path

from top10.race_selection import get_future_races

_TESTING_PATTERN = 'Pre-Season|Pre Season|Testing'


def load_f1_data(data_dir):
    """Load and filter test and training data CSVs."""
    data_dir = Path(data_dir)
    test_df = None
    training_df = None

    test_path = data_dir / 'test_data.csv'
    train_path = data_dir / 'training_data.csv'

    if test_path.exists():
        test_df = pd.read_csv(test_path)
        if 'EventName' in test_df.columns:
            test_df = test_df[
                ~test_df['EventName'].str.contains(_TESTING_PATTERN, case=False, na=False)
            ].copy()

    if train_path.exists():
        try:
            training_df = pd.read_csv(train_path)
            if 'EventName' in training_df.columns:
                training_df = training_df[
                    ~training_df['EventName'].str.contains(_TESTING_PATTERN, case=False, na=False)
                ].copy()
        except Exception:
            pass

    return test_df, training_df


def build_race_list(test_df):
    """Build the combined completed + future race list from test data.

    Returns a list of dicts with keys: year, eventName, roundNumber, isFuture.
    """
    completed_races = test_df[['Year', 'EventName', 'RoundNumber']].drop_duplicates()
    unique_years = sorted(test_df['Year'].unique())
    future_races_list = []

    for year in unique_years:
        try:
            future_races = get_future_races(year)
            if not future_races.empty:
                future_races_clean = future_races[['Year', 'EventName', 'RoundNumber']].drop_duplicates()
                future_only = future_races_clean[
                    ~future_races_clean.set_index(['Year', 'EventName', 'RoundNumber']).index.isin(
                        completed_races.set_index(['Year', 'EventName', 'RoundNumber']).index
                    )
                ]
                future_races_list.append(future_only)
        except Exception:
            pass

    all_races_df = completed_races.copy()
    if future_races_list:
        all_future = pd.concat(future_races_list, ignore_index=True)
        all_races_df = pd.concat([all_races_df, all_future], ignore_index=True)

    all_races_df = all_races_df.sort_values(['Year', 'RoundNumber'], ascending=[False, False])

    completed_index = completed_races.set_index(['Year', 'EventName', 'RoundNumber']).index
    races_list = []
    for _, race in all_races_df.iterrows():
        has_data = (race['Year'], race['EventName'], race['RoundNumber']) in completed_index
        races_list.append({
            'year': int(race['Year']),
            'eventName': race['EventName'],
            'roundNumber': int(race['RoundNumber']),
            'isFuture': not has_data,
        })

    return races_list


def get_filter_reason(row, is_future):
    """Return the filter reason string for a driver result, or None if not filtered."""
    if is_future:
        return None

    actual_pos = row.get('ActualPosition')
    if pd.isna(actual_pos):
        return None
    actual_pos = int(actual_pos)

    grid_pos = row.get('ActualGridPosition')
    if pd.isna(grid_pos):
        grid_pos = row.get('GridPosition')

    is_dnf = row.get('IsDNF', False)
    if pd.notna(is_dnf) and is_dnf:
        return 'DNF/DSQ'

    if pd.notna(grid_pos):
        position_drop = actual_pos - float(grid_pos)
        if position_drop > 6:
            return f'Dropped {int(position_drop)} places (grid {int(grid_pos)} → finish {actual_pos})'

    return None


def format_predictions(top10, is_future):
    """Format a top10 DataFrame into (predictions_filtered, predictions_unfiltered) lists."""
    predictions_filtered = []
    predictions_unfiltered = []

    for _, row in top10.iterrows():
        has_constructor = 'Constructor' in row.index
        actual_pos = row.get('ActualPosition')
        actual_position = int(actual_pos) if pd.notna(actual_pos) else None

        actual_grid_pos = row.get('ActualGridPosition')
        if pd.isna(actual_grid_pos):
            actual_grid_pos = row.get('GridPosition')
        display_grid_pos = int(actual_grid_pos) if pd.notna(actual_grid_pos) else None

        filter_reason = get_filter_reason(row, is_future)
        is_filtered = filter_reason is not None

        pred_data = {
            'rank': int(row.get('Rank', 0)),
            'driverName': str(row.get('DriverName', 'Unknown')),
            'driverNumber': int(row.get('DriverNumber', 0)) if pd.notna(row.get('DriverNumber')) else None,
            'predictedPosition': float(row['PredictedPosition']),
            'actualPosition': actual_position,
            'gridPosition': display_grid_pos,
            'constructor': str(row.get('Constructor', 'Unknown')) if has_constructor else None,
            'filterReason': filter_reason,
            'isFiltered': is_filtered,
        }

        predictions_unfiltered.append(pred_data)
        if not is_filtered:
            predictions_filtered.append(pred_data)

    return predictions_filtered, predictions_unfiltered
