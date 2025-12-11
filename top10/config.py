"""
Configuration for F1 prediction models.
"""

# Feature columns (9 features)
# SeasonPoints, SeasonStanding, SeasonAvgFinish, HistoricalTrackAvgPosition,
# ConstructorStanding, ConstructorTrackAvg, GridPosition, RecentForm, TrackType
FEATURE_COLS = ['SeasonPoints', 'SeasonStanding', 'SeasonAvgFinish', 'HistoricalTrackAvgPosition',
                'ConstructorStanding', 'ConstructorTrackAvg', 'GridPosition', 'RecentForm', 'TrackType']

# Training years: years to use for training data
TRAINING_YEARS = [2020, 2021, 2022, 2023, 2024]

# Test year: year to use for test data
TEST_YEAR = 2025

