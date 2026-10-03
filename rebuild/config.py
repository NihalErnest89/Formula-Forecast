from pathlib import Path

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / 'data'
OUT = Path(__file__).parent / 'saved'


FEATURE_COLS = [
    'SeasonPoints',
    'HistoricalTrackAvgPosition',
    'ConstructorStanding',
    'ConstructorTrackAvg',
    'GridPosition',
    'RecentForm',
    'CareerWins',
    'WinsLast3Years',
    'TrackType',
]

TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
TEST_YEARS =[2026]
HIDDEN = [64, 32]
BATCH_SIZE = 32
LR = 0.001
MAX_EPOCHS = 60
CV_SEEDS = [0, 1, 2]
