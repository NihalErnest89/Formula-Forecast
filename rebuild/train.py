import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from config import DEVICE, MIN_DELTA, PATIENCE, TOP10_WEIGHT, TRAIN_YEARS, FEATURE_COLS, HIDDEN, BATCH_SIZE, LR, MAX_EPOCHS, CV_SEEDS
from data import load_data, filter_races, split_years, split_train_val, fit_preprocessing, make_X
from model import build_model, permutation_importance, ranked_top10_mae, save_artifacts

# ---------------------------------------------------------------------------
# 5. dataloader
# ---------------------------------------------------------------------------
# float labels (not long) because this is regression: we want 3.7, not class 3.

def make_loader(X, y, w, shuffle):
    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y), torch.FloatTensor(w))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# ---------------------------------------------------------------------------
# 7. train loop
# ---------------------------------------------------------------------------
# each epoch: train on batches, then check validation.
# keep a snapshot of the best weights and restore it at the end --
# the last epoch is usually not the best one.
#
# with val_df=None it just trains for `epochs` epochs, no early stopping --
# that's how the final model trains on every season with nothing held back.

def train_model(model, X_train, y_train, epochs, val_df=None, X_val=None):
    model = model.to(DEVICE)
    w = np.where(y_train <= 10, TOP10_WEIGHT, 1.0)
    train_loader = make_loader(X_train, y_train, w, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    best_loss = float('inf')
    stale = 0

    curve = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            optimizer.zero_grad()
            loss = (wb * (model(xb).squeeze() - yb) ** 2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        if train_loss < best_loss - MIN_DELTA:
            best_loss = train_loss
            stale = 0
        else:
            stale += 1

        line = (f'    Epoch:{epoch + 1:3d} \tlr:{optimizer.param_groups[0]["lr"]:.4f}'
                f'\t train loss:{train_loss:.3f}')


        if val_df is not None:
            val_mae = ranked_top10_mae(model, val_df, X_val)
            curve.append(val_mae)
            line += f'\tval MAE:{val_mae:.3f}'
        if (epoch + 1) % 10 == 0 or (epoch + 1) == 1:
            print(line)
        scheduler.step()

        if stale >= PATIENCE:
            print(f'    Early stopping at epoch {epoch + 1}')
            break

    if val_df is not None and curve:
        curve += [curve[-1]] * (epochs - len(curve))
    return model, curve



# ---------------------------------------------------------------------------
# 3. cross-validation
# ---------------------------------------------------------------------------
# leave-one-season-out: each training season takes a turn as validation.
# the medians/scaler are refit and a fresh model is built INSIDE every fold --
# fit them once outside and each fold's validation season leaks into its own
# preprocessing.

def cross_validate(train_df):
    curves = []
    importances = []
    for val_year in TRAIN_YEARS:
        fold_train, fold_val = split_train_val(train_df, val_year)
        medians, scaler = fit_preprocessing(fold_train)
        X_tr = make_X(fold_train, medians, scaler)
        X_va = make_X(fold_val, medians, scaler)

        seed_curves = []
        for seed in CV_SEEDS:
            torch.manual_seed(1000 * seed + val_year)
            model = build_model(len(FEATURE_COLS), HIDDEN)
            _, curve = train_model(model, X_tr, fold_train['ActualPosition'].values, MAX_EPOCHS,
                                   val_df=fold_val, X_val=X_va)
            seed_curves.append(curve)
            if seed == CV_SEEDS[0]:
                importances.append(permutation_importance(model, fold_val, X_va))

        curves.append(np.mean(seed_curves, axis=0))
        print(f'  fold {val_year} done')
    return np.array(curves), np.mean(importances, axis=0)




# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    df = filter_races(load_data())
    train_df = split_years(df, TRAIN_YEARS)
    print(f'{len(train_df)} training rows across {TRAIN_YEARS}')
    print()

    print(f'cross-validating ({MAX_EPOCHS} epochs per fold)...')
    curves, importance = cross_validate(train_df)
    mean_curve = curves.mean(axis=0)
    best_epochs = int(mean_curve.argmin()) + 1

    print()
    print('mean CV top-10 MAE by epoch:')
    for e in range(10, MAX_EPOCHS + 1, 10):
        print(f'  epoch {e:3d}:  {mean_curve[e - 1]:.3f}')
    if best_epochs == MAX_EPOCHS:
        print('  warning: best epoch is the last one -- raise MAX_EPOCHS, it may still be improving')

    at_best = curves[:, best_epochs - 1]
    print()
    print(f'best epoch: {best_epochs}')
    for year, mae in zip(TRAIN_YEARS, at_best):
        print(f'  fold {year}:  top-10 MAE {mae:.3f}')
    print(f'CV top-10 MAE: {at_best.mean():.3f} +/- {at_best.std():.3f}   (folds range {at_best.min():.3f} - {at_best.max():.3f})')

    print()
    print('feature importance (extra top-10 error on held-out seasons when the feature is shuffled):')
    print(pd.Series(importance, index=FEATURE_COLS).sort_values(ascending=False).round(3).to_string())

    print()
    print(f'training final model on all {len(TRAIN_YEARS)} seasons for {best_epochs} epochs...')
    torch.manual_seed(0)
    medians, scaler = fit_preprocessing(train_df)
    model = build_model(len(FEATURE_COLS), HIDDEN)
    model, _ = train_model(model, make_X(train_df, medians, scaler),
                           train_df['ActualPosition'].values, best_epochs)
    save_artifacts(model, scaler, medians)



if __name__ == '__main__':
    main()
