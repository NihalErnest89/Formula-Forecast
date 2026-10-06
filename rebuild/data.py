import numpy as np
import pandas as pd
from config import FEATURE_COLS, DATA_DIR
from sklearn.preprocessing import StandardScaler



# ---------------------------------------------------------------------------
# 1. get the data
# ---------------------------------------------------------------------------
# the raw api data is cached in cache/ (fastf1 session cache), and
# collect_data.py already turned it into feature tables:
#   data/training_data.csv  -> 2018-2024, one row per driver per race
#   data/test_data.csv      -> 2025-2026

def load_data():
    training_data = pd.read_csv(DATA_DIR / 'training_data.csv')
    test_data = pd.read_csv(DATA_DIR / 'test_data.csv')
    return add_features(pd.concat([training_data, test_data], ignore_index=True))


# ---------------------------------------------------------------------------
# 1b. cleaned-up features
# ---------------------------------------------------------------------------
# computed HERE, on every row (retirements included), before filter_races drops
# anything -- PointsShare needs the leader even if the leader retired. every value
# only uses races BEFORE the one it describes.
#
#   PointsShare     season points / the leader's points (0-1). SeasonPoints grows
#                   0 -> ~500 through a season, so 60 pts means different things at
#                   round 3 and round 20; a share doesn't drift but keeps the gaps
#                   (standing loses them -- it tested worse).
#   RecentFormFin   mean finish over the last 5 races the driver FINISHED, this
#                   season (cars change every winter). a crash isn't pace, so
#                   retirements are skipped. round 1 falls back to last season.
#   TrackAvgShrunk  average finish at this circuit (renamed races merged, finishes
#                   only), pulled toward the driver's overall average when they've
#                   only been there once or twice -- one race isn't a track record.
#
# CV (8 seasons, real grid): the three together beat the old versions by 0.038,
# in 6 of 8 seasons.

# the same physical circuit raced under a different name
SAME_CIRCUIT = {
    '70th Anniversary Grand Prix': 'British Grand Prix',
    'Styrian Grand Prix': 'Austrian Grand Prix',
    'Mexican Grand Prix': 'Mexico City Grand Prix',
    'Brazilian Grand Prix': 'São Paulo Grand Prix',
    'Barcelona Grand Prix': 'Spanish Grand Prix',
}
SHRINK = 2.0   # how many "average" races the track average starts with


def circuit_of(event, year):
    if event == 'Spanish Grand Prix' and year >= 2026:   # 2026 Spanish GP is the new Madrid track
        return 'Madrid'
    return SAME_CIRCUIT.get(event, event)


def last_5_before(values, keys):
    """Mean of the last 5 non-missing values BEFORE each row, within each group."""
    def per_group(s):
        rolling = s.dropna().rolling(5, min_periods=1).mean()   # over finishes only
        return rolling.reindex(s.index).ffill().shift(1)         # carry forward, exclude today
    return values.groupby(keys, group_keys=False).apply(per_group)


def add_features(df):
    df = df.sort_values(['Year', 'RoundNumber']).reset_index(drop=True)
    finished = ~df['IsDNF'].astype(bool) & df['ActualPosition'].notna()
    finish = df['ActualPosition'].where(finished)          # NaN on a retirement

    # PointsShare
    points = df['SeasonPoints'].where(df['RoundNumber'] != 1, 0)
    leader = points.groupby([df['Year'], df['RoundNumber']]).transform('max')
    df['PointsShare'] = (points / leader.replace(0, np.nan)).fillna(0)

    # RecentFormFin
    this_season = last_5_before(finish, [df['DriverName'], df['Year']])
    last_season = last_5_before(finish, [df['DriverName']])
    df['RecentFormFin'] = this_season.fillna(last_season)

    # TrackAvgShrunk: running totals of finishes so far, per driver and per driver+circuit
    circuit = pd.Series([circuit_of(e, y) for e, y in zip(df['EventName'], df['Year'])], index=df.index)
    total = finish.fillna(0)
    count = finish.notna().astype(float)
    prior = lambda s: s.cumsum().shift(1)
    drv_total = total.groupby(df['DriverName']).transform(prior)
    drv_count = count.groupby(df['DriverName']).transform(prior)
    trk_total = total.groupby([df['DriverName'], circuit]).transform(prior).fillna(0)
    trk_count = count.groupby([df['DriverName'], circuit]).transform(prior).fillna(0)
    overall = drv_total / drv_count.replace(0, np.nan)    # NaN on debut -> median fill later
    df['TrackAvgShrunk'] = (trk_total + SHRINK * overall) / (trk_count + SHRINK)
    return df


# ---------------------------------------------------------------------------
# 2. filter the rows
# ---------------------------------------------------------------------------
# this is where most of the accuracy came from, not the model. drop:
#   - DNF / DSQ / DNS rows (IsDNF), can't score a position that doesn't exist
#   - outliers: finished 6+ places worse than they actually started
#   - anything outside the top 10 (ActualPosition > 10)
#
# then RENUMBER positions within each race. after filtering a race might read
# 1, 2, 4, 7 -- but the label has to be 1, 2, 3, 4.
#
# SeasonPoints is broken at round 1 (collect_data.py sums ALL previous seasons
# instead of just the last one), so zero it -- at round 1 nobody has scored yet.

def filter_races(df):
    df = df[~df['IsDNF']]
    df = df.copy()
    df = df.dropna(subset=['ActualPosition', 'ActualGridPosition'])
    df.loc[df['RoundNumber'] == 1, 'SeasonPoints'] = 0
    df['ActualPosition'] = df.groupby(['Year', 'RoundNumber'])['ActualPosition'].rank(method='first')
    return df


# ---------------------------------------------------------------------------
# 4. prepare features
# ---------------------------------------------------------------------------
# anything you FIT (medians, scaler) gets fit on TRAIN only, then applied to
# val/test. fitting on the eval sets leaks information.

def make_X(df, medians, scaler):
    """Turn a dataframe into a scaled feature matrix using ALREADY-FITTED
    medians and scaler. This is the only path features should ever take --
    training, evaluation, and prediction all go through here."""
    X = df[FEATURE_COLS].fillna(medians).values
    return scaler.transform(X)


def fit_preprocessing(train_df):
    medians = train_df[FEATURE_COLS].median()
    scaler = StandardScaler().fit(train_df[FEATURE_COLS].fillna(medians).values)
    return medians, scaler

def split_years(df, years):
    return df[df['Year'].isin(years)].copy()

def split_train_val(df, val_year):
    val_data = df[df['Year'] == val_year].copy()
    if val_data.empty:
        raise ValueError(f'no rows for val_year={val_year}; data has {sorted(df.Year.unique())}')
    train_data = df[df['Year'] != val_year].copy()
    return train_data, val_data