# Formula Forecast

Predicting Formula One finishing order with neural-network ensembles trained on FastF1 data.

**Live site:** https://NihalErnest89.github.io/Formula-Forecast

## Overview

Formula Forecast predicts the finishing order of F1 races. The models are trained on the
2018–2025 seasons and tested on the 2026 season, which they never see in training — the site
shows those 2026 predictions next to the real results.

There are two models, because what you know about a race changes once qualifying has happened:

- **Post-quali model** (used for completed races, and for the next race once qualifying is in):
  a 5-model ensemble that predicts how far each driver will finish *from their grid slot*, rather
  than their absolute position. Its predicted score is `grid rank + predicted change`, ranked within
  the race. 15 features per driver-race.
- **Pre-quali model** (used for upcoming races): the same idea, but starting from season form
  (average grid position) instead of the real grid, with walk-forward Elo ratings added.
  17 features per driver-race.

## Results

Measured on the **15 completed races of 2026** (a held-out season), using the model's predicted
top 10 for each race against where those drivers actually finished. The baseline is the simplest
possible prediction: *the race finishes in qualifying order*.

| | Model | Qualifying order |
|---|---|---|
| **Excluding DNFs and drops of more than 6 places** | | |
| Exact position | **38.2%** | 35.8% |
| Within 1 place | 75.6% | 75.6% |
| Within 2 places | 89.4% | 90.2% |
| Within 3 places | 95.9% | 95.9% |
| Average error (places) | 1.04 | 1.06 |
| **All predictions, including DNFs and crashes** | | |
| Exact position | 18.7% | 18.7% |
| Within 3 places | 68.0% | 68.0% |
| Average error (places) | 3.89 | 3.92 |

In plain terms: the model's predicted order is about as accurate as qualifying order, with a small
edge on exact positions. Qualifying order is a strong baseline — most of a race result is decided
by who starts where — and closing the remaining gap is the open problem this project works on.

## How it works

### Pipeline

One command refreshes everything and redeploys the site:

```bash
python update_frontend.py                 # collect -> train -> generate JSONs -> deploy
python update_frontend.py --skip-train    # reuse the existing models
python update_frontend.py --skip-deploy   # refresh data and predictions only
```

The steps it runs:

1. **`collect_data.py`** — pulls race results from the FastF1 API and builds one feature row per
   driver per race. Collection is incremental: per-season snapshots are kept in `data/raw/` and
   only newly completed races are fetched. Every feature for a race is computed only from races
   *before* it, so nothing leaks from the future.
2. **`top10/train.py`** — trains the models (uses CUDA when available).
3. **`generate_static_data.py`** — precomputes every race's predictions and the projected
   standings as JSON in `frontend/public/data/`. The site is fully static.
4. **`npm run deploy`** (in `frontend/`) — builds the React app and publishes it to GitHub Pages.

### Features

Base features (11), shared by both models:

| Feature | What it captures |
|---|---|
| `SeasonPoints`, `SeasonStanding` | championship position so far |
| `SeasonAvgFinish`, `RecentForm` | average finish this season / last 5 races |
| `HistoricalTrackAvgPosition` | driver's past results at this circuit |
| `ConstructorStanding`, `ConstructorTrackAvg` | car strength overall and at this circuit |
| `GridPosition` | the real qualifying grid slot (post-quali model), or the season-average grid as a stand-in before qualifying (pre-quali model) |
| `CareerWins`, `WinsLast3Years` | long-term and recent winning record |
| `TrackType` | street circuit vs permanent circuit |

The **post-quali model** adds `DriverAvgGain` and `ConstructorAvgGain` (average places gained
from grid position in recent races — racecraft), plus `OverQual` and `OverQualXCar` (how far
this weekend's real grid slot differs from the driver's usual one, and how that interacts with
car strength).

The **pre-quali model** adds the two racecraft features plus `DriverElo` and `ConstructorElo`
(walk-forward Elo ratings from race finishing order) and two track-affinity features.

### Model

- Feed-forward network: input → 64 → 32 → 1, each hidden layer Linear → BatchNorm → ReLU →
  Dropout (0.4), He initialisation.
- Trained on the change from grid order with a **Huber loss**, one race per training step.
  (The project started with MSE; it switched to Huber early on because DNFs and crashes produce
  huge errors that dominate a squared loss.)
- **5-seed ensemble** — five networks with different random seeds, predictions averaged.
- Adam optimiser with weight decay; early stopping on ranked error for the last training season.

### Evaluation

- **Held-out test season:** 2026 is never used for training or for choosing models.
- **Season-level cross-validation** on the training years (one fold per season). Note this is
  not strictly time-ordered — a fold can train on later seasons than the one it validates on —
  which is why the held-out 2026 season is the number that matters.
- Accuracy is always reported **next to the qualifying-order baseline**. A raw accuracy figure
  says little on its own when the grid already predicts most of the result.

## Project structure

```
.
├── collect_data.py           # FastF1 collection + feature engineering
├── update_frontend.py        # one-command refresh and deploy
├── generate_static_data.py   # precomputes the site's prediction JSONs
├── top10/                    # production models
│   ├── train.py              # trains all models
│   ├── config.py             # features, training/test years, ensemble seeds
│   ├── feature_calculation.py# racecraft, Elo and future-race features
│   ├── model_loader.py       # loads models and ensembles
│   └── predict.py            # prediction helpers
├── frontend/                 # React site (Predictions, Standings, Simulator, About)
├── api/                      # optional Flask API
├── rebuild/                  # from-scratch reimplementation (work in progress)
├── data/                     # feature tables (training_data.csv = 2018–2025, test_data.csv = 2026)
├── models/                   # trained models, scalers and ensemble metadata
└── top20/                    # legacy full-grid model
```

## Setup

```bash
pip install -r requirements.txt
cd frontend && npm install
```

### Changing the training / test years

The year split appears in three places, which must agree:

- `collect_data.py` → `main()` → `training_years` / `test_years`
- `top10/config.py` → `TRAINING_YEARS` / `TEST_YEARS`
- `rebuild/config.py` → `TRAIN_YEARS` / `TEST_YEARS`

After changing it, run `python collect_data.py --reorganize` — a plain run skips rebuilding the
feature tables when no new races have been fetched.

Years shown on the site are never used for training; otherwise the site would be grading the
model on races it has already seen.

## Dependencies

Key packages (see `requirements.txt`): `fastf1`, `pandas`, `numpy`, `torch`, `scikit-learn`,
`matplotlib`.
