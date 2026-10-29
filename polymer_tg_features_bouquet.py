# Does adding features on top of the RDKit descriptors help? Concatenates three
# blocks:
#   A) RDKit 2D descriptors (~200)  - from Tg_rdkit_descriptors.csv
#   B) hand-crafted ring features   - largest ring, fused bonds, etc.
#   C) Magpie elemental stats (132) - composition view via matminer
# Same bouquet + CV as polymer_tg_bouquet.py; prints a per-model delta vs the
# RDKit-only baseline. Short answer: doesn't move the top models - the ring info
# is already in RDKit and Magpie is redundant for these organic polymers. To
# actually beat it you'd need a different representation (fingerprints / GNN).
# Results -> polymer_tg_features_results.csv.

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

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
RDKIT_CACHE = HERE / 'Tg_rdkit_descriptors.csv'
RING_CACHE = HERE / 'Tg_ring_features.csv'
MAGPIE_CACHE = HERE / 'Tg_magpie_features.csv'
BASELINE_CSV = HERE / 'polymer_tg_model_results.csv'      # RDKit-only run
RESULTS_CSV = HERE / 'polymer_tg_features_results.csv'
SEED = 42

SCALE_SENSITIVE = {'Linear Regression', 'Ridge', 'Lasso', 'PLS Regression',
                   'KNN', 'SVR', 'MLP Regressor'}


# --- Block A: RDKit descriptors (reuse cache, else compute) ---
def rdkit_block(df):
    if RDKIT_CACHE.exists():
        X = pd.read_csv(RDKIT_CACHE, index_col=0)
        if len(X) == len(df):
            print(f'RDKit block (cached): {X.shape}')
            return X
    print('Computing RDKit descriptors ...')
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
        rows.append(vals); keep.append(df.index[i])
    X = pd.DataFrame(rows, columns=names, index=keep)
    X.to_csv(RDKIT_CACHE)
    return X


# --- Block B: extra ring / structural features ---
def ring_block(df):
    if RING_CACHE.exists():
        X = pd.read_csv(RING_CACHE, index_col=0)
        if len(X) == len(df):
            print(f'Ring block (cached): {X.shape}')
            return X
    print('Computing extra ring/structural features ...')
    rows, keep = [], []
    for idx, smi in zip(df.index, df['SMILES']):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        ri = mol.GetRingInfo()
        atom_rings = ri.AtomRings()
        bond_rings = ri.BondRings()
        heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 0]
        ring_atoms = set(a for r in atom_rings for a in r)
        hetero_in_ring = sum(1 for ai in ring_atoms
                             if mol.GetAtomWithIdx(ai).GetAtomicNum() not in (0, 6))
        # bonds shared by >1 ring (fused-ring indicator)
        from collections import Counter
        bond_counts = Counter(b for br in bond_rings for b in br)
        n_fused_bonds = sum(1 for _, c in bond_counts.items() if c > 1)
        n_arom_rings = sum(1 for r in atom_rings
                           if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r))
        smi_clean = smi.replace('*', '')
        rows.append(dict(
            has_ring=int(len(atom_rings) > 0),
            n_rings=len(atom_rings),
            largest_ring_size=max((len(r) for r in atom_rings), default=0),
            ring_atom_fraction=len(ring_atoms) / max(len(heavy), 1),
            n_heteroatoms_in_ring=hetero_in_ring,
            n_fused_ring_bonds=n_fused_bonds,
            n_aromatic_rings=n_arom_rings,
            n_aliphatic_rings=len(atom_rings) - n_arom_rings,
            n_branches=smi_clean.count('('),
            n_double_bonds=smi_clean.count('='),
            n_triple_bonds=smi_clean.count('#'),
        ))
        keep.append(idx)
    X = pd.DataFrame(rows, index=keep)
    X.to_csv(RING_CACHE)
    print(f'Ring block: {X.shape}')
    return X


# --- Block C: Magpie composition features ---
def smiles_to_composition(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    counts = {}
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 0:        # dummy '*' chain-end -> skip
            continue
        sym = a.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
        h = a.GetTotalNumHs()
        if h:
            counts['H'] = counts.get('H', 0) + h
    if not counts:
        return None
    return counts


def magpie_block(df):
    if MAGPIE_CACHE.exists():
        X = pd.read_csv(MAGPIE_CACHE, index_col=0)
        if len(X) == len(df):
            print(f'Magpie block (cached): {X.shape}')
            return X
    print('Computing Magpie elemental features (from SMILES composition) ...')
    ep = ElementProperty.from_preset('magpie')
    labels = ['Magpie_' + l for l in ep.feature_labels()]
    rows, keep = [], []
    for idx, smi in zip(df.index, df['SMILES']):
        comp = smiles_to_composition(smi)
        if comp is None:
            continue
        try:
            vals = ep.featurize(Composition(comp))
        except Exception:
            vals = [np.nan] * len(labels)
        rows.append(vals); keep.append(idx)
    X = pd.DataFrame(rows, columns=labels, index=keep)
    X.to_csv(MAGPIE_CACHE)
    print(f'Magpie block: {X.shape}')
    return X


def clean_features(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.mask(X.abs() > 1e12)
    X = X.dropna(axis=1, how='all')
    nunique = X.nunique()
    X = X[nunique[nunique > 1].index]
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
    steps = [SimpleImputer(strategy='median')]
    if name in SCALE_SENSITIVE:
        steps.append(StandardScaler())
    steps.append(model)
    return make_pipeline(*steps)


def main():
    df = pd.read_csv(DATA_CSV).dropna(subset=['SMILES', 'Tg', 'Polymer Class'])

    A = rdkit_block(df)
    B = ring_block(df)
    C = magpie_block(df)
    # align all blocks on the common parseable index
    idx = A.index.intersection(B.index).intersection(C.index)
    df = df.loc[idx]
    X_all = pd.concat([A.loc[idx], B.loc[idx], C.loc[idx]], axis=1)
    X_all = clean_features(X_all)
    print(f'\nCombined feature matrix: {X_all.shape[0]} polymers x {X_all.shape[1]} features '
          f'(RDKit {A.shape[1]} + ring {B.shape[1]} + Magpie {C.shape[1]}, pre-clean).\n')

    X = X_all.values
    y = df['Tg'].values
    groups = df['Polymer Class'].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    gkf = GroupKFold(n_splits=5)

    # baseline (RDKit-only) for the delta column
    base = None
    if BASELINE_CSV.exists():
        base = pd.read_csv(BASELINE_CSV).set_index('Model')['R2']

    results = []
    for name, model in build_models().items():
        pipe = wrap(name, model); pipe.fit(Xtr, ytr)
        pred = np.asarray(pipe.predict(Xte)).ravel()
        mae = mean_absolute_error(yte, pred)
        rmse = np.sqrt(mean_squared_error(yte, pred))
        r2 = r2_score(yte, pred)
        pred_grp = cross_val_predict(wrap(name, model), X, y, cv=gkf, groups=groups)
        r2_fam = r2_score(y, np.asarray(pred_grp).ravel())
        mae_fam = mean_absolute_error(y, np.asarray(pred_grp).ravel())
        delta = (r2 - base[name]) if base is not None and name in base else np.nan
        results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2,
                        'R2_rdkit_only': base[name] if base is not None and name in base else np.nan,
                        'dR2_vs_rdkit': delta,
                        'R2_family_holdout': r2_fam, 'MAE_family_holdout': mae_fam})
        print(f"{name:22s} R2={r2:.3f} (rdkit {base[name]:.3f}, d={delta:+.3f}) "
              f"| MAE={mae:5.1f} | family R2={r2_fam:.3f}"
              if base is not None and name in base else
              f"{name:22s} R2={r2:.3f} | MAE={mae:5.1f} | family R2={r2_fam:.3f}")

    res = pd.DataFrame(results).sort_values('R2', ascending=False)
    res.to_csv(RESULTS_CSV, index=False)

    print('\n=== Tg LEADERBOARD with RDKit + ring + Magpie features (random 80/20, by R2) ===')
    show = res[['Model', 'MAE', 'RMSE', 'R2', 'R2_rdkit_only', 'dR2_vs_rdkit', 'R2_family_holdout']]
    print(show.to_string(index=False,
          formatters={'MAE': '{:.1f}'.format, 'RMSE': '{:.1f}'.format, 'R2': '{:.3f}'.format,
                      'R2_rdkit_only': '{:.3f}'.format, 'dR2_vs_rdkit': '{:+.3f}'.format,
                      'R2_family_holdout': '{:.3f}'.format}))
    best = res.iloc[0]
    print(f"\n>>> Best (combined features): {best.Model}  R2={best.R2:.3f} "
          f"(RDKit-only was {best.R2_rdkit_only:.3f}, delta {best.dR2_vs_rdkit:+.3f}); "
          f"family-holdout R2={best.R2_family_holdout:.3f}")
    mean_delta = res['dR2_vs_rdkit'].mean()
    print(f"Mean R2 change across all models: {mean_delta:+.4f}")
    print(f'\nFull results -> {RESULTS_CSV}')


if __name__ == '__main__':
    main()
