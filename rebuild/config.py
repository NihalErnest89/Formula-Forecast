from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / 'data'
OUT = Path(__file__).parent / 'saved'

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')


FEATURE_COLS = [
    'ActualGridPosition',
    'PointsShare',
    'TrackAvgShrunk',
    'ConstructorStanding',
    'RecentFormFin',
    'TrackType'
]

TRAIN_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
TEST_YEARS =[2026]
HIDDEN = [64, 32]
BATCH_SIZE = 32
LR = 0.001
MAX_EPOCHS = 60
CV_SEEDS = [0, 1, 2]

PATIENCE = 5
MIN_DELTA = 0.01

DEFAULT_YEAR = 2026

TOP10_WEIGHT = 2.0
