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
    'ActualGridPosition',
    'RecentForm',
    'CareerWins',
    'WinsLast3Years',
    'TrackType',
]

VAL_YEAR = 2024
HIDDEN = [64, 32]
BATCH_SIZE = 32
LR = 0.001
MAX_EPOCHS = 200
MAX_PATIENCE = 20