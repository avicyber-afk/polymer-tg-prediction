# Try Morgan/ECFP fingerprints vs the RDKit descriptors. Fingerprints are a
# different view (2048 bits, each = presence of a circular substructure), so
# worth checking if they add anything the aggregate descriptors miss.
# Compares three feature sets - RDKit only, Morgan only, RDKit+Morgan - under
# both random KFold-5 and family-holdout GroupKFold-5, R2 mean +/- std.
# Upshot: combined edges RDKit a little on the random split (~0.90 vs 0.886),
# basically a wash on family-holdout. Results -> polymer_tg_fingerprint_results.csv.

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem, RDLogger
try:
    from rdkit.Chem import rdFingerprintGenerator
    _HAS_GEN = True
except Exception:
    from rdkit.Chem import AllChem
    _HAS_GEN = False

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold, cross_validate
from sklearn.linear_model import Ridge
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

HERE = Path(__file__).parent
DATA_CSV = HERE / 'Tg_SMILES_class_pid_polyinfo_median.csv'
RDKIT_CACHE = HERE / 'Tg_rdkit_descriptors.csv'
FP_CACHE = HERE / 'Tg_morgan_fp.npz'
RESULTS = HERE / 'polymer_tg_fingerprint_results.csv'
SEED = 42
N_BITS = 2048
RADIUS = 2


def morgan_matrix(smiles_list):
    if FP_CACHE.exists():
        d = np.load(FP_CACHE)
        if d['fp'].shape[0] == len(smiles_list):
            print(f'Morgan fingerprints (cached): {d["fp"].shape}')
            return d['fp'], d['ok']
    print(f'Computing Morgan fingerprints (radius {RADIUS}, {N_BITS} bits) ...')
    if _HAS_GEN:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    fps, ok = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(N_BITS, dtype=np.int8)); ok.append(False); continue
        if _HAS_GEN:
            arr = gen.GetFingerprintAsNumPy(mol).astype(np.int8)
        else:
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
            arr = np.frombuffer(bytes(bv.ToBitString(), 'ascii'), 'u1') - ord('0')
            arr = arr.astype(np.int8)
        fps.append(arr); ok.append(True)
    fp = np.vstack(fps).astype(np.int8)
    ok = np.array(ok)
    np.savez_compressed(FP_CACHE, fp=fp, ok=ok)
    print(f'Morgan fingerprints: {fp.shape} ({(~ok).sum()} unparseable rows zero-filled)')
    return fp, ok


def clean_descriptors(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.mask(X.abs() > 1e12)
    X = X.dropna(axis=1, how='all')
    nun = X.nunique()
    return X[nun[nun > 1].index]


def models():
    return {
        'Ridge': (Ridge(alpha=1.0, random_state=SEED), True),
        'Random Forest': (RandomForestRegressor(n_estimators=200, max_features='sqrt',
                                                random_state=SEED, n_jobs=-1), False),
        'Extra Trees': (ExtraTreesRegressor(n_estimators=200, max_features='sqrt',
                                            random_state=SEED, n_jobs=-1), False),
        'HistGradientBoosting': (HistGradientBoostingRegressor(random_state=SEED), False),
        'XGBoost': (XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=SEED, n_jobs=-1), False),
        'LightGBM': (LGBMRegressor(n_estimators=500, random_state=SEED, n_jobs=-1, verbose=-1), False),
    }


def wrap(model, scale):
    steps = [SimpleImputer(strategy='median')]
    if scale:
        steps.append(StandardScaler(with_mean=False))   # sparse-safe
    steps.append(model)
    return make_pipeline(*steps)


def main():
    df = pd.read_csv(DATA_CSV).dropna(subset=['SMILES', 'Tg', 'Polymer Class']).reset_index(drop=True)
    y = df['Tg'].values
    groups = df['Polymer Class'].values

    # RDKit descriptors (reuse cache; align by position -- cache was built on the
    # same cleaned CSV order)
    desc = pd.read_csv(RDKIT_CACHE, index_col=0).reset_index(drop=True)
    desc = clean_descriptors(desc)
    Xdesc = desc.values

    fp, ok = morgan_matrix(df['SMILES'].tolist())
    assert fp.shape[0] == len(df) == Xdesc.shape[0], (fp.shape, len(df), Xdesc.shape)

    feature_sets = {
        'RDKit only': Xdesc,
        'Morgan only': fp,
        'RDKit + Morgan': np.hstack([Xdesc, fp]),
    }
    print(f'\nRows: {len(df)}  | RDKit dims: {Xdesc.shape[1]}  | Morgan dims: {fp.shape[1]}\n')

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    gkf = GroupKFold(n_splits=5)
    rows = []
    for fs_name, X in feature_sets.items():
        print(f'--- feature set: {fs_name}  ({X.shape[1]} dims) ---')
        for m_name, (base, scale) in models().items():
            out = {'Features': fs_name, 'Model': m_name}
            for scheme, cv, grp in [('random', kf, None), ('family', gkf, groups)]:
                cvres = cross_validate(wrap(base, scale), X, y, cv=cv, groups=grp,
                                       scoring=['r2', 'neg_mean_absolute_error'], n_jobs=1)
                r2 = cvres['test_r2']; mae = -cvres['test_neg_mean_absolute_error']
                out[f'R2_{scheme}_mean'] = r2.mean(); out[f'R2_{scheme}_std'] = r2.std()
                out[f'MAE_{scheme}_mean'] = mae.mean()
            rows.append(out)
            print(f"  {m_name:22s} random R2={out['R2_random_mean']:.3f}+/-{out['R2_random_std']:.3f}"
                  f"  | family R2={out['R2_family_mean']:.3f}+/-{out['R2_family_std']:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS, index=False)

    print('\n=== BEST MODEL PER FEATURE SET (by random-CV R2) ===')
    for fs in feature_sets:
        t = res[res.Features == fs].sort_values('R2_random_mean').iloc[-1]
        print(f"  {fs:16s}: {t.Model:20s} random R2={t.R2_random_mean:.3f}+/-{t.R2_random_std:.3f}"
              f"  family R2={t.R2_family_mean:.3f}+/-{t.R2_family_std:.3f}")

    # headline comparison: best combined vs best RDKit-only
    b_rdkit = res[res.Features == 'RDKit only']['R2_random_mean'].max()
    b_comb = res[res.Features == 'RDKit + Morgan']['R2_random_mean'].max()
    b_fp = res[res.Features == 'Morgan only']['R2_random_mean'].max()
    print(f"\nBest random-CV R2:  RDKit={b_rdkit:.3f}  Morgan={b_fp:.3f}  Combined={b_comb:.3f}"
          f"  (combined vs RDKit dR2={b_comb-b_rdkit:+.3f})")
    print(f'\nFull results -> {RESULTS}')


if __name__ == '__main__':
    main()
