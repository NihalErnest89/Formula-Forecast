import json
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from config import BATCH_SIZE, FEATURE_COLS, HIDDEN, LR, MAX_EPOCHS, MAX_PATIENCE
from pathlib import Path
import pandas as pd
import json

DATA_DIR = Path(__file__).parent.parent / 'data'
OUT = Path(__file__).parent / 'saved'

# ---------------------------------------------------------------------------
# 5 + 6. dataloaders, model, loss, optimizer
# ---------------------------------------------------------------------------
# regression, not classification -- position 3 is closer to 4 than to 9.
# so: float labels, one output neuron, MSE loss.

def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


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
# 7. train loop
# ---------------------------------------------------------------------------
# each epoch: train on batches, then check validation.
# keep a snapshot of the best weights and restore it at the end --
# the last epoch is usually not the best one.

def train_model(model, X_train, y_train, X_val, y_val):
    train_loader = make_loader(X_train, y_train, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    best_val_mae = float('inf')
    best_state = None
    patience = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mae = (model(X_val_t).squeeze() - y_val_t).abs().mean().item()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= MAX_PATIENCE:
                print(f'  early stopping at epoch {epoch}')
                break

        if epoch % 10 == 0:
            print(f'  epoch {epoch:3d}   val MAE {val_mae:.3f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f'  best val MAE {best_val_mae:.3f}')
    return model, best_val_mae

# ---------------------------------------------------------------------------
# 8. evaluate
# ---------------------------------------------------------------------------
# the metric that matters is RANKED PER RACE: the model outputs raw scores and
# two drivers can both score 3.4 -- yet they can't both be P3. so rank within
# each race and compare those ranks to the true finish.
#
# always print the naive baseline next to it: just predicting the qualifying
# order. if you can't beat that, the model isn't doing anything.

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

    return df


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
    model.load_state_dict(torch.load(OUT / 'model.pth'))
    model.eval()

    # medians came back as a plain dict -- put it back in the shape fillna wants
    medians = pd.Series(meta['medians'])

    print(f'  loaded from {OUT}')
    return model, scaler, medians
