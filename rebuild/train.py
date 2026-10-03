import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from config import TRAIN_YEARS, FEATURE_COLS, HIDDEN, BATCH_SIZE, LR, MAX_EPOCHS, CV_SEEDS
from data import load_data, filter_races, split_years, split_train_val, fit_preprocessing, make_X
from model import build_model, ranked_top10_mae, save_artifacts

# ---------------------------------------------------------------------------
# 5. dataloader
# ---------------------------------------------------------------------------
# float labels (not long) because this is regression: we want 3.7, not class 3.

def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
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
    train_loader = make_loader(X_train, y_train, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    curve = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()

        if val_df is not None:
            curve.append(ranked_top10_mae(model, val_df, X_val))

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

        curves.append(np.mean(seed_curves, axis=0))
        print(f'  fold {val_year} done')
    return np.array(curves)




# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    df = filter_races(load_data())
    train_df = split_years(df, TRAIN_YEARS)
    print(f'{len(train_df)} training rows across {TRAIN_YEARS}')
    print()

    print(f'cross-validating ({MAX_EPOCHS} epochs per fold)...')
    curves = cross_validate(train_df)
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
    print(f'training final model on all {len(TRAIN_YEARS)} seasons for {best_epochs} epochs...')
    torch.manual_seed(0)
    medians, scaler = fit_preprocessing(train_df)
    model = build_model(len(FEATURE_COLS), HIDDEN)
    model, _ = train_model(model, make_X(train_df, medians, scaler),
                           train_df['ActualPosition'].values, best_epochs)
    save_artifacts(model, scaler, medians)



if __name__ == '__main__':
    main()
