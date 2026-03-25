"""
Configuration for F1 prediction models.
"""

# Feature columns (11 features)
# + WinsLast3Years: wins in last 3 calendar years (recency / on fire)
FEATURE_COLS = [
    'SeasonPoints',
    'SeasonStanding',
    'SeasonAvgFinish',
    'HistoricalTrackAvgPosition',
    'ConstructorStanding',
    'ConstructorTrackAvg',
    'GridPosition',
    'RecentForm',
    'CareerWins',
    'WinsLast3Years',
    'TrackType',
]

# Training years: years to use for training data (exclude years in TEST_YEARS)
TRAINING_YEARS = [2020, 2021, 2022, 2023, 2024]

# Test years: years to use for test data (predict will offer races from these years)
TEST_YEARS = [2025, 2026]

# Backward compat: if train.py only supported single TEST_YEAR, it uses TEST_YEARS[0] when applicable
TEST_YEAR = None  # Unused when TEST_YEARS is set; kept for reference

# If True, train 10-feature (no WinsLast3Years) vs 11-feature (with WinsLast3Years) and compare test accuracy
RUN_CAREER_WINS_COMPARISON = False

