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
    return pd.concat([training_data, test_data], ignore_index=True)


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
    df['ActualGridPosition'] = df['ActualGridPosition'].replace(0, 22)
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