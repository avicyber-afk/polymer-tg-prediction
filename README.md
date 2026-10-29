# polymer-tg-prediction

Predicting polymer **glass-transition temperature (Tg)** from SMILES — RDKit descriptors + Morgan fingerprints, a 15-model benchmark, and scaffold/family-holdout evaluation for honest out-of-distribution metrics.

Dataset: 7,208 polymers with SMILES, polymer class, and median Tg (PolyInfo-derived), included as `Tg_SMILES_class_pid_polyinfo_median.csv`.

## Headline results

Best model is XGBoost. Representation matters more than adding feature blocks:

| Feature set | random-CV R² | family-holdout R² |
|---|---|---|
| RDKit descriptors (~200) | 0.886 ± 0.005 | 0.771 ± 0.036 |
| Morgan fingerprints (2048) | 0.890 ± 0.008 | 0.757 ± 0.042 |
| **RDKit + Morgan** | **0.900 ± 0.004** | 0.782 ± 0.039 |

Adding Magpie composition features on top of RDKit did **not** help (within noise) — the descriptor set was already near its ceiling; a different *representation* (fingerprints) gave a small real gain, mostly on the random split.

Key insight: a **family-holdout** split (grouping by polymer class) exposes a ~35% MAE increase on unseen chemistries that a naive random split hides — the honest generalization metric.

## Repository layout

| File | What it does |
|---|---|
| `Tg_SMILES_class_pid_polyinfo_median.csv` | Source dataset: SMILES, polymer class, Tg (7,208 rows) |
| `polymer_tg_bouquet.py` | 15-model bouquet on RDKit descriptors; random + family-holdout CV |
| `polymer_tg_features_bouquet.py` | Adds hand-crafted ring features + Magpie composition features (negative result) |
| `polymer_tg_fingerprints.py` | **Main representation study** — RDKit vs Morgan vs combined, with error bars |
| `polymer_tg_insights.py` | Class-wise Tg landscape, structure-property drivers, family-holdout gap |
| `polymer_tg_*_results.csv`, `polymer_tg_class_stats.csv` | Result tables |
| `polymer_tg_insights.png` | Multi-panel insights figure |

## Reproduce

```bash
pip install -r requirements.txt

# RDKit-descriptor bouquet (caches descriptors on first run)
python polymer_tg_bouquet.py

# Representation study: RDKit vs Morgan fingerprints vs combined
python polymer_tg_fingerprints.py

# Feature-augmentation experiment (ring + Magpie)
python polymer_tg_features_bouquet.py

# Non-objective insights
python polymer_tg_insights.py
```

## Notes

- Feature caches (`Tg_rdkit_descriptors.csv`, `Tg_morgan_fp.npz`, `Tg_magpie_features.csv`, `Tg_ring_features.csv`) are **gitignored** — the scripts regenerate them from the source CSV on first run.
- Every score is reported as R² mean ± std across 5 CV folds, with preprocessing fit inside each fold (no leakage).
- `polymer_tg_features_bouquet.py` needs `matminer` + `pymatgen` (for Magpie); the others need only `rdkit` + scikit-learn/boosting libs.
