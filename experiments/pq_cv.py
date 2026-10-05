"""Season-level CV for the PRODUCTION post-quali delta model, scored the way
the site scores it. Tests rebuild-derived changes one at a time.

Configs (cumulative):
  C0  production as-is
  C1  + pit-lane grid 0 -> back of grid
  C2  + train on all finishers (not just actual top-10, no drop filter)
  C3  + epoch count from the season-averaged held-out curve (no early stopping
        on 2025-in-training)
  C4  + SeasonPoints zeroed at round 1

Site metric (identical for every config): rank ALL drivers by
grid_rank + mean(ensemble delta), take the top 10, drop DNFs and finishers more
than 6 places behind their (corrected) grid slot, re-rank model / actual /
qualifying order among the rest, compare. 2026 is never touched.
"""
import sys, time, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler

ROOT = r'c:/Users/nerne/Documents/Nihal/College/UC Santa Cruz/Grad/Year 2/CSE 244a/Need-For-Predictions'
sys.path.insert(0, ROOT + '/top10')
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from train import F1NeuralNetwork, train_postquali_delta, build_delta_races
    from feature_calculation import add_racecraft_features, add_overqual_features
    from config import FEATURE_COLS_POSTQUALI

SEEDS = [42, 43, 44]
CURVE_EPOCHS = 60
OUT = ROOT + '/json/pq_cv_results.json'


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


raw = pd.read_csv(ROOT + '/data/training_data.csv')          # 2018-2025 only
raw = raw[raw['RoundNumber'] >= 1].reset_index(drop=True)
YEARS = sorted(raw['Year'].unique())
field = raw.groupby(['Year', 'RoundNumber'])['DriverNumber'].transform('count')
# Evaluation grid is ALWAYS the corrected one, so the filter is identical across configs.
raw['EvalGrid'] = raw['ActualGridPosition'].where(raw['ActualGridPosition'] > 0, field)


def prep(pitfix, zero_r1):
    d = raw.copy()
    if zero_r1:
        d.loc[d['RoundNumber'] == 1, 'SeasonPoints'] = 0
    if pitfix:
        d['ActualGridPosition'] = d['EvalGrid']
    d['SeasonAvgGrid'] = d['GridPosition']
    d['GridPosition'] = d['ActualGridPosition'].fillna(d['GridPosition'])
    quiet(add_racecraft_features, [d])
    add_overqual_features(d)
    return d


def races_allfinishers(df, feats, med):
    out = []
    for (yr, rnd), g in df.groupby(['Year', 'RoundNumber']):
        g = g[~g['IsDNF'].fillna(False)].dropna(subset=['ActualPosition', 'GridPosition']).copy()
        if len(g) < 5:
            continue
        X = g[feats].copy()
        for c in feats:
            X[c] = X[c].fillna(med[c])
        out.append({'year': yr, 'X': X.values.astype(np.float32),
                    'fin': g['ActualPosition'].rank(method='first').values.astype(np.float32),
                    'base': g['GridPosition'].rank(method='first').values.astype(np.float32)})
    return out


def eval_races(df, feats, med):
    """Held-out races in site form: every driver of the race."""
    out = []
    for (yr, rnd), g in df.groupby(['Year', 'RoundNumber']):
        X = g[feats].copy()
        for c in feats:
            X[c] = X[c].fillna(med[c])
        out.append({'X': X.values.astype(np.float32),
                    'base': g['GridPosition'].rank(method='first').values,
                    'act': g['ActualPosition'].values.astype(float),
                    'egrid': g['EvalGrid'].values.astype(float),
                    'dnf': g['IsDNF'].fillna(False).values.astype(bool)})
    return out


def site_errors(races, deltas):
    em, eq = [], []
    for r, dl in zip(races, deltas):
        top = np.argsort(r['base'] + dl, kind='stable')[:10]
        keep = [i for i in top if not r['dnf'][i] and not np.isnan(r['act'][i])
                and r['act'][i] - r['egrid'][i] <= 6]
        if len(keep) < 2:
            continue
        keep = np.array(keep)
        pos_in_top = {i: k for k, i in enumerate(top)}
        pr = pd.Series([pos_in_top[i] for i in keep]).rank(method='first').values
        ar = pd.Series(r['act'][keep]).rank(method='first').values
        qr = pd.Series(r['egrid'][keep]).rank(method='first').values
        em += list(np.abs(pr - ar)); eq += list(np.abs(qr - ar))
    return np.array(em), np.array(eq)


def predict_deltas(models, races, scaler):
    out = []
    with torch.no_grad():
        for r in races:
            X = torch.FloatTensor(scaler.transform(r['X']))
            out.append(np.mean([m(X).numpy() for m in models], axis=0))
    return out


def train_curve(races_tr, races_va, n_in, scaler, seed, held, epochs):
    """Production training loop, but no early stopping: record the held-out
    site error every epoch. LR schedule still follows production's internal
    validation races, so only the stopping rule changes."""
    from train import ranked_eval_delta
    torch.manual_seed(seed); np.random.seed(seed)
    m = quiet(F1NeuralNetwork, input_size=n_in, hidden_sizes=[64, 32], dropout_rate=0.4)
    opt = optim.Adam(m.parameters(), lr=0.003, weight_decay=2e-4)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10, min_lr=1e-5)
    huber = nn.HuberLoss(delta=1.0)
    order = np.arange(len(races_tr)); curve = []
    for ep in range(epochs):
        m.train(); np.random.shuffle(order)
        for i in order:
            r = races_tr[i]
            X = torch.FloatTensor(scaler.transform(r['X']))
            loss = huber(m(X), torch.FloatTensor(r['fin']) - torch.FloatTensor(r['base']))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step(ranked_eval_delta(m, races_va, scaler)['mae'])
        m.eval()
        em, _ = site_errors(held, predict_deltas([m], held, scaler))
        curve.append(em.mean())
    return np.array(curve)


def run(name, pitfix, allfin, curve, zero_r1):
    t0 = time.time()
    d = prep(pitfix, zero_r1)
    feats = [f for f in FEATURE_COLS_POSTQUALI if f in d.columns]
    folds = []
    curves = []
    for vy in YEARS:
        trd, ted = d[d['Year'] != vy], d[d['Year'] == vy]
        med = {c: trd[c].median() for c in feats}
        races_tr = races_allfinishers(trd, feats, med) if allfin else build_delta_races(trd, feats, med)
        races_va = [r for r in races_tr if r['year'] == trd['Year'].max()]   # production's (in-training) val
        sc = StandardScaler().fit(np.vstack([r['X'] for r in races_tr]))
        held = eval_races(ted, feats, med)
        if curve:
            curves.append(np.mean([train_curve(races_tr, races_va, len(feats), sc, s, held, CURVE_EPOCHS)
                                   for s in SEEDS], axis=0))
            folds.append(None)
        else:
            models = [quiet(train_postquali_delta, races_tr, races_va, len(feats), sc, seed=s) for s in SEEDS]
            em, eq = site_errors(held, predict_deltas(models, held, sc))
            folds.append((em.mean(), eq.mean(), (em == 0).mean() * 100, (eq == 0).mean() * 100))
        print(f'    {name}: fold {vy} done ({time.time() - t0:.0f}s)', flush=True)
    res = {'name': name}
    if curve:
        cv = np.array(curves); mean_curve = cv.mean(0); best = int(mean_curve.argmin())
        res['best_epoch'] = best + 1
        res['fold_mae'] = cv[:, best].tolist()
        res['curve_every10'] = [round(float(mean_curve[e]), 4) for e in range(9, CURVE_EPOCHS, 10)]
        # quali baseline is model-independent given the fixed filter? No: the
        # filtered set depends on the model's top 10. Recompute at best epoch
        # is not possible without models; record from the non-curve run instead.
    else:
        res['fold_mae'] = [f[0] for f in folds]; res['fold_quali'] = [f[1] for f in folds]
        res['fold_exact'] = [f[2] for f in folds]; res['fold_quali_exact'] = [f[3] for f in folds]
    print(f'  {name}: CV site MAE {np.mean(res["fold_mae"]):.4f}'
          + (f'  (quali {np.mean(res["fold_quali"]):.4f}, exact {np.mean(res["fold_exact"]):.1f}% vs {np.mean(res["fold_quali_exact"]):.1f}%)'
             if not curve else f'  best epoch {res["best_epoch"]}') + f'  [{time.time() - t0:.0f}s]', flush=True)
    return res


if __name__ == '__main__':
    configs = [
        ('C0 production', dict(pitfix=False, allfin=False, curve=False, zero_r1=False)),
        ('C1 +pitlane', dict(pitfix=True, allfin=False, curve=False, zero_r1=False)),
        ('C2 +allfinishers', dict(pitfix=True, allfin=True, curve=False, zero_r1=False)),
        ('C3 +curve-epochs', dict(pitfix=True, allfin=True, curve=True, zero_r1=False)),
        ('C4 +zeroR1', dict(pitfix=True, allfin=True, curve=False, zero_r1=True)),
    ]
    only = sys.argv[1:]
    results = []
    for name, kw in configs:
        if only and not any(name.startswith(o) for o in only):
            continue
        results.append(run(name, **kw))
        json.dump(results, open(OUT, 'w'), indent=1)
    print('\nfold years:', YEARS)
    for r in results:
        print(r['name'], [round(x, 3) for x in r['fold_mae']])
