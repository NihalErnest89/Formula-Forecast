"""
Generate the static JSON files for the GitHub Pages site from the rebuild/ models.

Run after collecting data and training (python rebuild/train.py):
    python generate_static_data.py

Races shown on the site: every completed race of the test season + the NEXT race only
(further races would get near-identical predictions - nothing new is known about them).
  completed race  -> post-quali model (real starting grid)
  next race       -> post-quali if its qualifying has been saved by collect_data.py,
                     otherwise pre-quali (a projection from season form)
The Standings page still projects the whole season: each remaining race is predicted
behind the scenes with the pre-quali model, but only the next race gets a page.

Outputs:
    frontend/public/data/races.json
    frontend/public/data/predictions/<year>-<roundNumber>.json
    frontend/public/data/standings/<year>.json
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'rebuild'))
import data as rb                                   # noqa: E402  rebuild/data.py
from config import MODELS, TEST_YEARS, DATA_DIR      # noqa: E402  rebuild/config.py
from model import load_artifacts                    # noqa: E402  rebuild/model.py

OUT_DIR = ROOT / 'frontend' / 'public' / 'data'

# F1 points system (race only; future sprint points are not modeled)
POINTS_BY_RANK = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}


def _canon_driver(x):
    """Normalize driver number to a string key ('4', 4.0 and 4 are the same driver)."""
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        return str(x)


# ---------------------------------------------------------------------------
# prediction file format (unchanged from before, so the frontend needs no changes)
# ---------------------------------------------------------------------------

def filter_reason(row, is_future):
    """Why a driver is hidden in the site's filtered view, or None."""
    if is_future or pd.isna(row.get('ActualPosition')):
        return None
    if bool(row.get('IsDNF', False)):
        return 'DNF/DSQ'
    actual = int(row['ActualPosition'])
    grid = row.get('ActualGridPosition')
    if pd.notna(grid) and actual - float(grid) > 6:
        return f'Dropped {int(actual - grid)} places (grid {int(grid)} → finish {actual})'
    return None


def format_predictions(top10, is_future):
    filtered, unfiltered = [], []
    for _, row in top10.iterrows():
        grid = row.get('ActualGridPosition')
        if pd.isna(grid):
            grid = row.get('GridPosition')        # future race: the projected grid
        reason = filter_reason(row, is_future)
        p = {
            'rank': int(row['Rank']),
            'driverName': str(row['DriverName']),
            'driverNumber': int(row['DriverNumber']) if pd.notna(row.get('DriverNumber')) else None,
            'predictedPosition': float(row['PredictedPosition']),
            'actualPosition': int(row['ActualPosition']) if pd.notna(row.get('ActualPosition')) else None,
            'gridPosition': int(round(grid)) if pd.notna(grid) else None,
            'constructor': None,
            'filterReason': reason,
            'isFiltered': reason is not None,
        }
        unfiltered.append(p)
        if reason is None:
            filtered.append(p)
    return filtered, unfiltered


def predict_race(rows, name, loaded, is_future):
    """Rank every driver in the race with one model and build the race's JSON."""
    model, scaler, medians = loaded[name]
    feats = MODELS[name]['features']
    with torch.no_grad():
        scores = model(torch.FloatTensor(rb.make_X(rows, medians, scaler, feats))).squeeze(-1).numpy()
    ranked = rows.assign(PredictedPosition=scores).sort_values('PredictedPosition')
    ranked['Rank'] = range(1, len(ranked) + 1)
    if name == 'prequali':
        ranked['GridPosition'] = ranked['ProjectedGrid']     # shown as the grid before qualifying
    filtered, unfiltered = format_predictions(ranked.head(10), is_future)
    first = rows.iloc[0]
    return {
        'race': {
            'year': int(first['Year']),
            'eventName': str(first['EventName']),
            'roundNumber': int(first['RoundNumber']),
            'isFuture': is_future,
            'usedQualiGrid': name == 'postquali',
            'modelUsed': name,
        },
        'predictionsFiltered': filtered,
        'predictionsUnfiltered': unfiltered,
        'totalDrivers': len(rows),
    }


# ---------------------------------------------------------------------------
# standings
# ---------------------------------------------------------------------------

def generate_standings(year, test_df, predictions):
    """
    Championship standings for a season, from the per-race predictions.
    - predictedPoints: model points for FUTURE races -> projectedTotal/Rank
    - modelPoints: model points for COMPLETED races -> modelRank
    """
    season = test_df[test_df['Year'] == year].copy()
    if season.empty:
        return None
    season['_dn'] = season['DriverNumber'].apply(_canon_driver)

    drivers = {}
    for dn, grp in season.groupby('_dn'):
        latest = grp.sort_values('RoundNumber').iloc[-1]
        drivers[dn] = {'driverNumber': dn, 'driverName': latest.get('DriverName', 'UNK'),
                       'teamName': latest.get('TeamName', ''),
                       'currentPoints': float(grp['Points'].fillna(0).sum()),
                       'predictedPoints': 0.0, 'modelPoints': 0.0}

    n_future = 0
    for pred in predictions:
        is_future = pred['race']['isFuture']
        n_future += int(is_future)
        for p in pred['predictionsFiltered']:
            pts = POINTS_BY_RANK.get(p.get('rank'))
            if pts is None:
                continue
            dn = _canon_driver(p.get('driverNumber'))
            entry = drivers.setdefault(dn, {'driverNumber': dn, 'driverName': p.get('driverName', 'UNK'),
                                            'teamName': '', 'currentPoints': 0.0,
                                            'predictedPoints': 0.0, 'modelPoints': 0.0})
            entry['predictedPoints' if is_future else 'modelPoints'] += pts

    rows = list(drivers.values())
    for r in rows:
        r['projectedTotal'] = r['currentPoints'] + r['predictedPoints']
    for key, rank_field in (('currentPoints', 'currentRank'), ('projectedTotal', 'projectedRank'),
                            ('modelPoints', 'modelRank')):
        for i, r in enumerate(sorted(rows, key=lambda x: -x[key]), 1):
            r[rank_field] = i
    rows.sort(key=lambda r: -r['projectedTotal'])

    return {
        'year': int(year),
        'completedRounds': int(season['RoundNumber'].nunique()),
        'remainingRounds': n_future,
        'seasonComplete': n_future == 0,
        'note': 'Projected totals = actual points from completed races + model-predicted points '
                '(25-18-15-...-1) for each remaining race (pre-qualifying projections, as if the '
                'season continues as it is). Model points = the same points system applied to the '
                "model's predictions of completed races. Sprint points are not modeled.",
        'standings': rows,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    year = max(TEST_YEARS)
    print('Loading models...')
    loaded = {name: load_artifacts(spec) for name, spec in MODELS.items()}

    test_df = pd.read_csv(DATA_DIR / 'test_data.csv')
    base = pd.concat([pd.read_csv(DATA_DIR / 'training_data.csv'), test_df], ignore_index=True)
    done = sorted(int(r) for r in test_df.loc[test_df['Year'] == year, 'RoundNumber'].unique())
    schedule = json.loads((DATA_DIR / f'schedule_{year}.json').read_text(encoding='utf-8'))
    remaining = sorted((s['RoundNumber'], s['EventName']) for s in schedule
                       if s['Year'] == year and s['RoundNumber'] not in done)

    # every remaining round gets placeholder rows (state after the latest race) before
    # the features are computed; only the next one can carry a saved qualifying grid
    placeholders = [rb.placeholder_rows(base, year, rnd, event) for rnd, event in remaining]
    rows = rb.add_features(pd.concat([base] + placeholders, ignore_index=True))
    season = rows[rows['Year'] == year]

    races, predictions = [], []
    for rnd in done:
        predictions.append(predict_race(season[season['RoundNumber'] == rnd], 'postquali', loaded, False))
    for i, (rnd, event) in enumerate(remaining):
        r = season[season['RoundNumber'] == rnd]
        name = 'postquali' if (i == 0 and r['ActualGridPosition'].notna().all()) else 'prequali'
        predictions.append(predict_race(r, name, loaded, True))

    # pages: completed races + the next race (newest first, like before)
    shown = [p for p in predictions if not p['race']['isFuture']] + \
            [p for p in predictions if p['race']['isFuture']][:1]
    shown.sort(key=lambda p: -p['race']['roundNumber'])
    next_round = remaining[0][0] if remaining else None

    predictions_dir = OUT_DIR / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    keep = set()
    for p in shown:
        race = p['race']
        name = f"{race['year']}-{race['roundNumber']}.json"
        (predictions_dir / name).write_text(json.dumps(p), encoding='utf-8')
        keep.add(name)
        races.append({'year': race['year'], 'eventName': race['eventName'], 'roundNumber': race['roundNumber'],
                      'isFuture': race['isFuture'], 'isNextRound': race['roundNumber'] == next_round,
                      'model': race['modelUsed']})
    stale = [f for f in predictions_dir.glob('*.json') if f.name not in keep]
    for f in stale:
        f.unlink()
    (OUT_DIR / 'races.json').write_text(json.dumps({'races': races}), encoding='utf-8')
    print(f"Wrote {len(shown)} race pages ({len(done)} completed + next race), removed {len(stale)} old ones")
    if remaining:
        print(f"  next race: R{next_round} {remaining[0][1]} -> {shown[0]['race']['modelUsed']} model")

    standings_dir = OUT_DIR / 'standings'
    standings_dir.mkdir(exist_ok=True)
    standings = generate_standings(year, test_df, predictions)
    (standings_dir / f'{year}.json').write_text(json.dumps(standings), encoding='utf-8')
    leader = standings['standings'][0]
    print(f"  Standings {year}: {standings['completedRounds']} done + {standings['remainingRounds']} projected; "
          f"projected champion: {leader['driverName']} ({leader['projectedTotal']:.0f} pts)")
    print(f'Done. Output: {OUT_DIR}')


if __name__ == '__main__':
    main()
