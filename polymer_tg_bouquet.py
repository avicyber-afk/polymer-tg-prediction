# Model bouquet for Tg. SMILES -> RDKit descriptors (~200) -> regression.
# The '*' chain-end wildcards parse as dummy atoms; unparseable mols get
# dropped, all-NaN/constant descriptor columns dropped, rest median-imputed.
# Linear/KNN/SVR/MLP get a StandardScaler, trees use raw features.
# Scored two ways: random 80/20, and family-holdout (GroupKFold by polymer
# class) for unseen-chemistry generalisation. XGBoost wins (~0.90 / ~0.79).
# Results -> polymer_tg_model_results.csv.

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

HERE = Path(__file__).parent
DATA_CSV = HERE / 'Tg_SMILES_class_pid_polyinfo_median.csv'
DESC_CACHE = HERE / 'Tg_rdkit_descriptors.csv'
RESULTS_CSV = HERE / 'polymer_tg_model_results.csv'
SEED = 42

SCALE_SENSITIVE = {'Linear Regression', 'Ridge', 'Lasso', 'PLS Regression',
                   'KNN', 'SVR', 'MLP Regressor'}


def compute_descriptors(df):
    if DESC_CACHE.exists():
        cached = pd.read_csv(DESC_CACHE, index_col=0)
        print(f'Loaded cached descriptors: {cached.shape}')
        return cached
    print('Computing RDKit descriptors for', len(df), 'polymers ...')
    names = [n for n, _ in Descriptors._descList]
    rows, keep = [], []
    for i, smi in enumerate(df['SMILES']):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        vals = []
        for _, fn in Descriptors._descList:
            try:
                vals.append(fn(mol))
            except Exception:
                vals.append(np.nan)
        rows.append(vals)
        keep.append(df.index[i])
    X = pd.DataFrame(rows, columns=names, index=keep)
    X.to_csv(DESC_CACHE)
    print(f'Descriptors computed: {X.shape} ({len(df)-len(X)} SMILES unparseable, dropped)')
    return X


def clean_features(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    # A handful of RDKit descriptors (notably Ipc, the Bonchev-Trinajstic
    # information content) grow factorially with molecule size and overflow
    # float32 -- they poison OLS/tree models. Treat any pathologically large
    # magnitude as missing so it gets median-imputed rather than dominating.
    X = X.mask(X.abs() > 1e12)
    X = X.dropna(axis=1, how='all')                 # drop all-NaN descriptors
    nunique = X.nunique()
    X = X[nunique[nunique > 1].index]               # drop constant descriptors
    return X


def build_models():
    return {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=SEED),
        'Lasso': Lasso(alpha=0.1, random_state=SEED, max_iter=10000),
        'PLS Regression': PLSRegression(n_components=10),
        'KNN': KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
        'SVR': SVR(C=100, gamma='scale'),
        'Decision Tree': DecisionTreeRegressor(max_depth=12, random_state=SEED),
        'Random Forest': RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        'Extra Trees': ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, random_state=SEED),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=SEED),
        'XGBoost': XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=SEED, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=500, random_state=SEED, n_jobs=-1, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=500, random_state=SEED, verbose=0),
        'MLP Regressor': MLPRegressor(hidden_layer_sizes=(256, 128), activation='relu',
                                      solver='adam', max_iter=500, random_state=SEED,
                                      early_stopping=True),
    }


def wrap(name, model):
    """Impute NaNs for every model; add scaling for scale-sensitive ones."""
    steps = [SimpleImputer(strategy='median')]
    if name in SCALE_SENSITIVE:
        steps.append(StandardScaler())
    steps.append(model)
    return make_pipeline(*steps)


def main():
    df = pd.read_csv(DATA_CSV).dropna(subset=['SMILES', 'Tg', 'Polymer Class'])
    Xdesc = compute_descriptors(df)
    df = df.loc[Xdesc.index]                        # align to parseable rows
    Xdesc = clean_features(Xdesc)
    print(f'Modelling on {Xdesc.shape[0]} polymers x {Xdesc.shape[1]} descriptors.\n')

    X = Xdesc.values
    y = df['Tg'].values
    groups = df['Polymer Class'].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

    gkf = GroupKFold(n_splits=5)
    results = []
    for name, base in build_models().items():
        # random 80/20 hold-out
        pipe = wrap(name, base)
        pipe.fit(Xtr, ytr)
        pred = np.asarray(pipe.predict(Xte)).ravel()
        mae = mean_absolute_error(yte, pred)
        rmse = np.sqrt(mean_squared_error(yte, pred))
        r2 = r2_score(yte, pred)

        # family-holdout R2 (fresh pipeline via CV)
        pred_grp = cross_val_predict(wrap(name, base), X, y, cv=gkf, groups=groups)
        r2_fam = r2_score(y, np.asarray(pred_grp).ravel())
        mae_fam = mean_absolute_error(y, np.asarray(pred_grp).ravel())

        results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2,
                        'R2_family_holdout': r2_fam, 'MAE_family_holdout': mae_fam})
        print(f"{name:22s} [random] MAE={mae:5.1f} RMSE={rmse:5.1f} R2={r2:.3f} "
              f"| [family] R2={r2_fam:.3f} MAE={mae_fam:5.1f}")

    res = pd.DataFrame(results).sort_values('R2', ascending=False)
    res.to_csv(RESULTS_CSV, index=False)

    print('\n=== Tg LEADERBOARD (random 80/20 split, by R2) ===')
    print(res.to_string(index=False))
    best = res.iloc[0]
    print(f"\n>>> Best Tg model: {best.Model}  "
          f"(random R2={best.R2:.3f}, RMSE={best.RMSE:.1f}C, MAE={best.MAE:.1f}C; "
          f"family-holdout R2={best.R2_family_holdout:.3f})")
    print(f'\nFull results -> {RESULTS_CSV}')


if __name__ == '__main__':
    main()
