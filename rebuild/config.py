from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / 'data'
OUT = Path(__file__).parent / 'saved'

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')


# two models: everything that differs between them lives here, so the rest of
# the code just picks one by name ('postquali' or 'prequali')
POSTQUALI_FEATURES = ['ActualGridPosition', 'PointsShare', 'TrackAvgShrunk',
                      'ConstructorStanding', 'RecentFormFin', 'TrackType']
PREQUALI_FEATURES = ['ProjectedGrid' if c == 'ActualGridPosition' else c
                     for c in POSTQUALI_FEATURES]

MODELS = {
    'postquali': {'features': POSTQUALI_FEATURES, 'grid': 'ActualGridPosition', 'dir': OUT / 'postquali'},
    'prequali':  {'features': PREQUALI_FEATURES,  'grid': 'ProjectedGrid',      'dir': OUT / 'prequali'},
}


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
