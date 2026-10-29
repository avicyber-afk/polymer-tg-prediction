# Some exploratory analysis on the Tg data, mostly using the Polymer Class
# column:
#   1. Tg by family - which polymer classes run hot vs cold
#   2. family generalisation gap - family-holdout vs random split (how well a
#      model does on a chemistry it never saw)
#   3. cheap regex SMILES descriptors (length, ring/aromatic counts) vs Tg
#   4. coverage - Tg range and class imbalance
# Uses plain regex descriptors so it runs without RDKit.
# Writes polymer_tg_insights_summary.csv, polymer_tg_class_stats.csv, .png.

import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

HERE = Path(__file__).parent
DATA_CSV = HERE / 'Tg_SMILES_class_pid_polyinfo_median.csv'
SUMMARY_CSV = HERE / 'polymer_tg_insights_summary.csv'
CLASS_CSV = HERE / 'polymer_tg_class_stats.csv'
OUT_PNG = HERE / 'polymer_tg_insights.png'


def smiles_descriptors(smiles: str) -> dict:
    """RDKit-free structural descriptors from a SMILES string."""
    s = smiles.replace('*', '')  # strip polymer chain-end wildcards
    aromatic = len(re.findall(r'[cnops]', s))          # lowercase = aromatic atoms
    rings = len(re.findall(r'[1-9]', s)) // 2           # ring-closure digit pairs
    return dict(
        length=len(s),
        n_C=len(re.findall(r'[Cc]', s)),
        n_O=len(re.findall(r'[Oo]', s)),
        n_N=len(re.findall(r'[Nn]', s)),
        n_aromatic=aromatic,
        n_rings=rings,
        n_double=s.count('='),
        n_triple=s.count('#'),
        n_branch=s.count('('),
        frac_aromatic=aromatic / max(len(s), 1),
    )


def load():
    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=['SMILES', 'Tg', 'Polymer Class']).copy()
    desc = pd.DataFrame([smiles_descriptors(s) for s in df['SMILES']], index=df.index)
    return pd.concat([df, desc], axis=1), list(desc.columns)


# ---------------------------------------------------------------------------
# Insight 1: class-wise Tg landscape
# ---------------------------------------------------------------------------
def insight_class_landscape(df):
    print("\n[Insight 1] Class-wise Tg landscape (families that run hot vs cold)")
    g = df.groupby('Polymer Class')['Tg'].agg(['count', 'mean', 'std', 'min', 'max'])
    g = g[g['count'] >= 30].sort_values('mean')
    print("  Coldest families (lowest mean Tg):")
    for cls, r in g.head(4).iterrows():
        print(f"    {cls[:45]:45s} mean={r['mean']:6.1f}C  std={r['std']:5.1f}  n={int(r['count'])}")
    print("  Hottest families (highest mean Tg):")
    for cls, r in g.tail(4).iloc[::-1].iterrows():
        print(f"    {cls[:45]:45s} mean={r['mean']:6.1f}C  std={r['std']:5.1f}  n={int(r['count'])}")
    g.to_csv(CLASS_CSV)
    print(f"  Full per-class table -> {CLASS_CSV.name}")
    return g


# ---------------------------------------------------------------------------
# Insight 2: family-holdout generalisation gap
# ---------------------------------------------------------------------------
def insight_family_gap(df, feat_cols):
    print("\n[Insight 2] Polymer-family generalisation gap (random split vs family holdout)")
    X = df[feat_cols].values
    y = df['Tg'].values
    groups = df['Polymer Class'].values

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

    # Random 5-fold
    pred_rand = cross_val_predict(rf, X, y, cv=KFold(5, shuffle=True, random_state=42))
    r2_rand = r2_score(y, pred_rand)
    mae_rand = mean_absolute_error(y, pred_rand)

    # Group 5-fold (never train & test on the same polymer family)
    pred_grp = cross_val_predict(rf, X, y, cv=GroupKFold(5), groups=groups)
    r2_grp = r2_score(y, pred_grp)
    mae_grp = mean_absolute_error(y, pred_grp)

    print(f"  Random split:  R2={r2_rand:.3f}  MAE={mae_rand:5.1f}C")
    print(f"  Family holdout: R2={r2_grp:.3f}  MAE={mae_grp:5.1f}C")
    print(f"  --> Generalisation gap: MAE rises {mae_grp-mae_rand:+.1f}C "
          f"({(mae_grp/mae_rand-1)*100:+.0f}%) on unseen families; "
          f"R2 drops {r2_rand-r2_grp:.3f}.")
    print("     (random-split R2 overstates accuracy on genuinely new chemistries)")
    return dict(r2_random=r2_rand, mae_random=mae_rand,
                r2_family_holdout=r2_grp, mae_family_holdout=mae_grp)


# ---------------------------------------------------------------------------
# Insight 3: structural-driver signal (RDKit-free)
# ---------------------------------------------------------------------------
def insight_structure_signal(df, feat_cols):
    print("\n[Insight 3] Structural drivers of Tg (RDKit-free SMILES descriptors)")
    corrs = {c: np.corrcoef(df[c], df['Tg'])[0, 1] for c in feat_cols}
    for c, v in sorted(corrs.items(), key=lambda kv: -abs(kv[1]))[:6]:
        print(f"    r(Tg, {c:14s}) = {v:+.3f}")
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(df[feat_cols], df['Tg'])
    imp = sorted(zip(feat_cols, rf.feature_importances_), key=lambda kv: -kv[1])
    print("  Top RF feature importances: " +
          ", ".join(f"{k}={v:.2f}" for k, v in imp[:4]))
    return corrs


# ---------------------------------------------------------------------------
# Insight 4: coverage / applicability domain
# ---------------------------------------------------------------------------
def insight_coverage(df):
    print("\n[Insight 4] Coverage / applicability domain")
    print(f"  Tg spans {df['Tg'].min():.0f} to {df['Tg'].max():.0f} C "
          f"(median {df['Tg'].median():.0f}C) across {df['Polymer Class'].nunique()} classes.")
    vc = df['Polymer Class'].value_counts()
    tail = (vc < 30).sum()
    print(f"  Class imbalance: largest family n={vc.iloc[0]}, "
          f"{tail} of {len(vc)} families have <30 samples "
          f"(model will be weak on those).")
    return dict(tg_min=float(df['Tg'].min()), tg_max=float(df['Tg'].max()),
                n_classes=int(df['Polymer Class'].nunique()),
                n_small_classes=int(tail))


def make_figure(df, class_stats, feat_cols):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    top = class_stats.tail(12)
    ax[0, 0].barh([c[:30] for c in top.index], top['mean'],
                  xerr=top['std'], color='teal')
    ax[0, 0].set(title='Mean Tg by polymer family (top 12)', xlabel='Tg (C)')

    ax[0, 1].hist(df['Tg'], bins=50, color='slateblue', edgecolor='k')
    ax[0, 1].set(title='Tg distribution (all polymers)', xlabel='Tg (C)', ylabel='count')

    ax[1, 0].scatter(df['n_aromatic'], df['Tg'], alpha=0.15, s=8, color='crimson')
    ax[1, 0].set(title='Aromatic-atom count vs Tg', xlabel='# aromatic atoms', ylabel='Tg (C)')

    ax[1, 1].scatter(df['frac_aromatic'], df['Tg'], alpha=0.15, s=8, color='darkgreen')
    ax[1, 1].set(title='Aromatic fraction vs Tg', xlabel='aromatic fraction', ylabel='Tg (C)')

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    print(f"\nFigure -> {OUT_PNG}")


def main():
    df, feat_cols = load()
    print(f"Loaded {len(df)} polymers, {df['Polymer Class'].nunique()} classes, "
          f"{len(feat_cols)} SMILES descriptors.")

    class_stats = insight_class_landscape(df)
    gap = insight_family_gap(df, feat_cols)
    insight_structure_signal(df, feat_cols)
    cov = insight_coverage(df)

    summary = pd.DataFrame([
        dict(Insight='family_gap', **gap),
        dict(Insight='coverage', **cov),
    ])
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSummary -> {SUMMARY_CSV}")

    make_figure(df, class_stats, feat_cols)


if __name__ == '__main__':
    main()
