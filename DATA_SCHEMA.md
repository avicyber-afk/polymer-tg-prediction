# Data schema — polymer-tg-prediction

This documents every dataset the pipeline reads or produces. Files marked
**(gitignored)** are not in the repo — the scripts regenerate them from the
source CSV on first run.

---

## 1. Source dataset — `Tg_SMILES_class_pid_polyinfo_median.csv` **(in repo, 840 KB)**

Grain: one row per **polymer**. 7,208 rows × 4 columns. PK = `PID`.
(Note: the file has a UTF-8 BOM before the first header.)

| Column | Type | Unit | Description |
|---|---|---|---|
| `SMILES` | string | — | Repeat-unit structure in SMILES; `*` marks chain-end attachment points (7,174 unique). The sole model input. |
| `PID` | string | — | PolyInfo polymer id (source key; unique per row) |
| `Polymer Class` | string | — | Chemical family, often a comma-joined multi-label (198 unique). Used for the family-holdout CV grouping. |
| `Tg` | float | °C | Glass-transition temperature (median value). Prediction target; range −139 to 495 °C. |

~34 SMILES appear more than once (same structure, different measured Tg).

---

## 2. Feature caches **(all gitignored)** — regenerated from the source CSV

These are numeric feature matrices derived from `SMILES`, one row per polymer
(aligned to the source CSV order). Rebuilt automatically on first script run.

| File | Shape | Produced by | Contents |
|---|---|---|---|
| `Tg_rdkit_descriptors.csv` | 7208 × 217 | `polymer_tg_bouquet.py` | RDKit 2D molecular descriptors (`MolWt`, `TPSA`, `NumAromaticRings`, `FractionCSP3`, …). ~198 survive cleaning (all-NaN / constant / overflow columns like `Ipc` dropped). |
| `Tg_morgan_fp.npz` | 7208 × 2048 | `polymer_tg_fingerprints.py` | Morgan/ECFP circular fingerprints (radius 2, 2048 bits), `int8`. Keys: `fp` (matrix), `ok` (parse mask). |
| `Tg_ring_features.csv` | 7208 × 11 | `polymer_tg_features_bouquet.py` | Hand-crafted ring/structural features (ring count, largest ring, fused-ring bonds, aromatic-ring count, branch/bond counts). |
| `Tg_magpie_features.csv` | 7208 × 132 | `polymer_tg_features_bouquet.py` | Magpie elemental-statistics features (min/max/range/mean/avg-dev/mode over ~22 elemental properties), from the repeat-unit composition. |

Descriptors are per-molecule — no cross-row leakage.

---

## 3. Result tables (in repo)

**`polymer_tg_model_results.csv`** — RDKit-descriptor bouquet (15 models), °C.

| Column | Description |
|---|---|
| `Model` | Model name |
| `MAE`, `RMSE`, `R2` | Metrics on a random 80/20 hold-out |
| `R2_family_holdout`, `MAE_family_holdout` | GroupKFold-by-class (family-holdout) metrics |

**`polymer_tg_fingerprint_results.csv`** — representation study (RDKit vs Morgan vs combined), R² mean ± std.

| Column | Description |
|---|---|
| `Features` | `RDKit only`, `Morgan only`, `RDKit + Morgan` |
| `Model` | Model name |
| `R2_random_mean`, `R2_random_std`, `MAE_random_mean` | Random 5-fold CV |
| `R2_family_mean`, `R2_family_std`, `MAE_family_mean` | Family-holdout 5-fold CV |

**`polymer_tg_features_results.csv`** — ring + Magpie augmentation experiment.

| Column | Description |
|---|---|
| `Model` | Model name |
| `MAE`, `RMSE`, `R2` | Combined-feature random-split metrics |
| `R2_rdkit_only` | Baseline R² (RDKit only) for the same model |
| `dR2_vs_rdkit` | R² change vs the RDKit-only baseline |
| `R2_family_holdout`, `MAE_family_holdout` | Family-holdout metrics |

**`polymer_tg_class_stats.csv`** — per-class Tg aggregates.

| Column | Description |
|---|---|
| `Polymer Class` | Chemical family |
| `count`, `mean`, `std`, `min`, `max` | Tg statistics for that family (classes with ≥30 samples) |

**`polymer_tg_insights_summary.csv`** — single-row summary of the insights run.

| Column | Description |
|---|---|
| `Insight` | `family_gap` / `coverage` marker |
| `r2_random`, `mae_random` | Random-split RF metrics |
| `r2_family_holdout`, `mae_family_holdout` | Family-holdout RF metrics (the generalization gap) |
| `tg_min`, `tg_max` | Tg range |
| `n_classes`, `n_small_classes` | Number of polymer classes / classes with <30 samples |
