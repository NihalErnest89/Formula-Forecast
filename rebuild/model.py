import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from config import DEVICE, FEATURE_COLS, HIDDEN, OUT

# ---------------------------------------------------------------------------
# 6. the model
# ---------------------------------------------------------------------------
# regression, not classification -- position 3 is closer to 4 than to 9.
# so: one output neuron that predicts a position directly.

def build_model(input_size, hidden):
    layers = []
    prev = input_size
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# 8. evaluate
# ---------------------------------------------------------------------------
# the metric that matters is RANKED PER RACE: the model outputs raw scores and
# two drivers can both score 3.4 -- yet they can't both be P3. so rank within
# each race and compare those ranks to the true finish.
#
# always print the naive baseline next to it: just ranking by season-average
# grid. if you can't beat that, the model isn't doing anything.
#
# ranked_top10_mae and evaluate compute the same top-10 number -- kept side by
# side so a change to the metric can't land in one and not the other.

def ranked_top10_mae(model, df, X):
    model.eval()
    with torch.no_grad():
        p = model(torch.FloatTensor(X).to(DEVICE)).squeeze().cpu().numpy()

    d = df.copy()
    d['p'] = p
    d['r'] = d.groupby(['Year', 'RoundNumber'])['p'].rank(method='first')
    top = d[d['ActualPosition'] <= 10]
    return (top['r'] - top['ActualPosition']).abs().mean()


def permutation_importance(model, df, X, repeats=5):
    """How much worse the held-out error gets when one feature is scrambled.

    Shuffle a single column (breaking its link to each driver while leaving the
    others intact), re-measure ranked_top10_mae, and report the rise in error.
    Big rise = the model leans on that feature; ~0 or negative = it ignores it
    (or it was adding noise)."""
    base = ranked_top10_mae(model, df, X)
    rng = np.random.default_rng(0)
    importance = []
    for j in range(X.shape[1]):
        rises = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            rises.append(ranked_top10_mae(model, df, Xp) - base)
        importance.append(np.mean(rises))
    return np.array(importance)


def evaluate(model, test_data, X_test):
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).squeeze().numpy()

    df = test_data.copy()
    df['pred'] = preds
    df['pred_rank'] = df.groupby(['Year', 'RoundNumber'])['pred'].rank(method='first')
    df['grid_rank'] = df.groupby(['Year', 'RoundNumber'])['ActualGridPosition'].rank(method='first')

    df['err'] = (df['pred_rank'] - df['ActualPosition']).abs()
    df['gerr'] = (df['grid_rank'] - df['ActualPosition']).abs()

    def report(label, sub):
        e, g = sub['err'], sub['gerr']
        print(f'  {label} (n={len(sub)})')
        print(f'    model  MAE {e.mean():.3f}  exact {(e==0).mean()*100:.1f}%  within-1 {(e<=1).mean()*100:.1f}%')
        print(f'    grid   MAE {g.mean():.3f}  exact {(g==0).mean()*100:.1f}%  within-1 {(g<=1).mean()*100:.1f}%')
        d = e.mean() - g.mean()
        print(f'    -> {"beats" if d < 0 else "LOSES to"} grid by {abs(d):.3f}')

    report('full field', df)
    report('true top 10', df[df['ActualPosition'] <= 10])

    model_picks = df[df['pred_rank'] <= 10]['err']
    grid_picks = df[df['grid_rank'] <= 10]['gerr']
    print(f'  predicted top 10 (what the site shows)')
    print(f'    model  MAE {model_picks.mean():.3f}  exact {(model_picks==0).mean()*100:.1f}%  within-1 {(model_picks<=1).mean()*100:.1f}%')
    print(f'    grid   MAE {grid_picks.mean():.3f}  exact {(grid_picks==0).mean()*100:.1f}%  within-1 {(grid_picks<=1).mean()*100:.1f}%')
    
    d = model_picks.mean() - grid_picks.mean()
    print(f'    -> {"beats" if d < 0 else "LOSES to"} grid by {abs(d):.3f}')

    return df


def show_weights(model, feature_cols):
    w = model[0].weight.detach().abs().mean(dim=0)
    s = pd.Series(w / w.sum(), index=feature_cols)
    print(s.sort_values(ascending=False).to_string())


# ---------------------------------------------------------------------------
# 9. save / load the model
# ---------------------------------------------------------------------------
# save the state_dict AND the scaler AND the medians. a model without the
# things that were fitted alongside it is useless -- you'd feed it differently
# scaled features at predict time and get garbage. the feature list and hidden
# sizes go in too, so load_artifacts can rebuild the exact architecture.

def save_artifacts(model, scaler, medians):
    OUT.mkdir(exist_ok=True)

    torch.save(model.state_dict(), OUT / 'model.pth')

    with open(OUT / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open(OUT / 'meta.json', 'w') as f:
        json.dump({
            'features': FEATURE_COLS,
            'medians': medians.to_dict(),
            'hidden': HIDDEN,
        }, f, indent=2)

    print(f'  saved to {OUT}')


def load_artifacts():
    with open(OUT / 'meta.json') as f:
        meta = json.load(f)

    with open(OUT / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    model = build_model(len(meta['features']), meta['hidden'])
    model.load_state_dict(torch.load(OUT / 'model.pth', map_location=DEVICE))
    model.eval()

    # medians came back as a plain dict -- put it back in the shape fillna wants
    medians = pd.Series(meta['medians'])

    print(f'  loaded from {OUT}')
    return model, scaler, medians
