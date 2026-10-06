# Changelog

Scores below are 8-season cross-validation (train on 7 of 2018–2025, test on the
8th, repeat), top-10 error in places — lower is better. A change only counts if it
wins in most of the 8 seasons; differences under ~0.02 are noise.

## 2026-10-06

### Rebuild (`rebuild/`)
- **Top-10 finishers count double in training** (`TOP10_WEIGHT = 2.0`, weighted
  squared error in `train_model`). Every driver still trains — unlike the old
  "top-10 only" approach, which threw the rest away. Better on all three scores
  with no midfield cost; beats qualifying order on the predicted-top-10 score in
  8/8 seasons (was 6/8). **CV 1.788 → 1.780.**
- **Site-style score added to `evaluate`** ("predicted top 10"): grades the drivers
  each method *put* in its top 10, the view a site visitor sees. The existing
  "true top 10" score picks who to grade using the result, which hides
  confident-but-wrong picks.

### Experiments
| idea | result |
|---|---|
| Predict change from the grid (delta) instead of position | slightly worse on the site score (−0.159 vs −0.181 vs grid); the real grid as an input already gives the "stays where they started" default |
| Top-10 weight ×2 / ×3 (with delta) | no help on top of delta |
| `TrackAvgShrunk` anchor: career vs this season vs last 20 finishes | career best — a steady baseline; "current car" is already covered by other features |

### Rebuild vs production, 2026 (16 races, identical scoring)
| | full field | predicted top 10 (site view) | exact, predicted top 10 |
|---|---|---|---|
| rebuild | 1.827 (−0.180 vs grid) | 1.794 (+0.012) | 24.4% |
| production | 1.942 (−0.065) | 1.794 (+0.012) | 28.7% |
| qualifying order | 2.007 | 1.781 | 30.0% |

Rebuild is far better through the field, tied with production on the site's top-10
view (7–6–3 race by race), and behind it on exact positions at the front. 2026's
front of the grid was unusually processional (qualifying order 30% exact vs ~25% in
training seasons). One season is noisy, so no site change yet.

## 2026-10-05

### Data fixes (`collect_data.py`, production feature code)
- **Pre-2018 wins now counted.** Results data starts in 2018, so career wins
  ignored everything earlier (Hamilton showed 44, not 106). `fetch_pre2018_wins.py`
  pulls 1950–2017 winners once from Jolpica into `data/pre2018_wins.json`;
  cross-checked against each driver's career total: 0 mismatches.
  Retrained and deployed. *(commit d79f681)*
- **Retirement flag fixed.** The DNF check searched the status for "DNF"/"DSQ"/"NC",
  but FastF1 writes "Retired", "Collision", "Engine"…, so only 22 of ~590
  retirements were flagged and ~490 crashes were treated as last-place finishes.
  New `_is_dnf` helper: anything other than "Finished" / "Lapped" / "+N Lap" is a
  retirement. Data regenerated. *(uncommitted; production site models not yet
  retrained on it)*

### Rebuild (`rebuild/`)
- **Feature importance** (`permutation_importance` in `model.py`), printed after
  training and in `predict.py`: shuffle one feature, measure how much worse the
  held-out error gets.
- **Real grid instead of season-average grid** (`ActualGridPosition`), and the
  "just use the grid" baseline in `evaluate` switched to match. Model now beats
  qualifying order by ~0.27 places in all 8 seasons (on the actual-top-10 metric).
- **Features removed:** `CareerWins`, `ConstructorTrackAvg`, `WinsLast3Years`
  (no help; the 3-year wins count rated drivers on wins from the previous rules
  era — e.g. Verstappen 17 vs Antonelli 8 going into 2026 Bahrain).
- **Optimizer:** `AdamW` with its default weight decay (L2 regularization).
- **Three cleaned-up features** added to `data.py` (`add_features`, run inside
  `load_data`):
  - `PointsShare` — points ÷ leader's points (replaces `SeasonPoints`, which grows
    all season)
  - `RecentFormFin` — last-5 average over races actually finished, resets each
    season (replaces `RecentForm`, where one crash counted as ~P19)
  - `TrackAvgShrunk` — track average with renamed circuits merged, crashes
    skipped, pulled toward the driver's usual result until they have history
    there (replaces `HistoricalTrackAvgPosition`)
  - Together: **1.827 → 1.788, better in 6/8 seasons.** To use: swap the three
    names into `FEATURE_COLS` in `config.py`.

### Experiments (`experiments/`) — what was tested and the verdict
| idea | result |
|---|---|
| Qualifying gap to pole / pre-penalty quali position | no gain (the real grid already carries it) |
| Points vs standing vs points-per-race | standing worse (loses gaps); share of leader slightly best |
| Recent-form window (3/8/12 races, weighted) | no difference — **resetting each season** is what matters |
| Wins this season / last season / 3 years / none | all within noise; removing wins costs nothing |
| Career wins with real grid | no gain |
| Track character (power / high-speed / twisty / balanced) | worse; model fits patterns that don't carry to a new season |
| Overtaking difficulty per circuit (grid-vs-finish) | no gain for the ranking — better suited to the site's simulator randomness |
| Weight decay (Adam 1e-4/1e-3, AdamW 0.01/0.1) | all tied, slightly better than none |
| Dropout 0.2 | worse in 8/8 seasons |

## 2026-10-04

### Production pipeline (deployed)
- **Driver history keyed on identity, not car number.** Numbers change and get
  reused (Verstappen 33→1→3, Norris 4→1, Ricciardo's #3 → Verstappen), so wins,
  form and track history followed the number: 2026 Verstappen showed 3 career wins
  (Ricciardo's), Norris 53 (Verstappen's). Fixed in `collect_data.py`,
  `top10/feature_calculation.py`, `generate_static_data.py`. *(commit 0010a22)*
- **Calendar check** (`top10/data_utils.py`): FastF1's schedule source returned
  403 and fell back to a shorter, differently numbered calendar, dropping the
  site's upcoming races. A fetched calendar is now only trusted if it matches the
  races already run; otherwise `data/schedule_2026.json` is used.
- **Qualifying lookup guard** (`generate_static_data.py`): with the degraded
  calendar, asking for Singapore's qualifying silently returned Hungary's. A
  mismatched event is now rejected.
- **Faster data refresh** (`collect_data.py`): when only current-season races are
  new, the training-season rows are reused instead of rebuilt — 161 s → 20 s,
  output verified identical.
- Site: retrained models deployed; 2026 lead over qualifying order unchanged
  (avg error 1.000 vs 1.031, exact 39.2% vs 36.9%).

### Rebuild (`rebuild/`)
- `predict.py`: race list shows names (R1 Australian GP …); menu to train /
  predict / collect new races (`collect_races` calls `collect_data.main()` from the
  project root); prints every feature next to each prediction.
- `train.py`: per-epoch printout (learning rate, train loss, validation error);
  learning-rate decay (`ExponentialLR`, ×0.95 per epoch); stops when training loss
  stops improving (`PATIENCE`, `MIN_DELTA`) — saves time; the epoch count itself
  is still picked from the cross-validation curve.
- Training runs on CPU: measured ~2× faster than GPU for a network this small.
