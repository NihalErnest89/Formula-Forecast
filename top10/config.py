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

# Post-quali model features: base features (GridPosition = ACTUAL quali grid)
# plus racecraft features that predict deviations from the qualifying order.
# The post-quali model is trained on a DELTA target (finish - grid rank) so it
# starts from the quali order and learns corrections - this beats both the raw
# model and naive quali order (see json/beat_the_grid.json).
FEATURE_COLS_POSTQUALI = FEATURE_COLS + [
    'DriverAvgGain',        # driver's mean (grid - finish) over last 20 races
    'ConstructorAvgGain',   # constructor's mean gain over last 40 entries
    'OverQual',             # actual grid - season-avg grid (penalty/over-performance signal)
    'OverQualXCar',         # recovery cross: starting far back x car strength
]

# Number of seeds in the post-quali ensemble (scores averaged before ranking;
# confirmed to improve 2026 accuracy vs single seed)
POSTQUALI_ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]

# Pre-quali delta model: base = season-average-grid (form) order, target =
# deviation from it. Elo ratings help HERE (no quali info exists yet) even
# though they add nothing post-quali. See json/phase2_prequali.json.
FEATURE_COLS_PREQUALI = FEATURE_COLS + [
    'DriverAvgGain',
    'ConstructorAvgGain',
    'DriverElo',
    'ConstructorElo',
]
PREQUALI_ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]

# Training years: years to use for training data (exclude years in TEST_YEARS)
TRAINING_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Test years: years to use for test data (predict will offer races from these years)
TEST_YEARS = [2025, 2026]

# Backward compat: if train.py only supported single TEST_YEAR, it uses TEST_YEARS[0] when applicable
TEST_YEAR = None  # Unused when TEST_YEARS is set; kept for reference

# If True, train 10-feature (no WinsLast3Years) vs 11-feature (with WinsLast3Years) and compare test accuracy
RUN_CAREER_WINS_COMPARISON = False

