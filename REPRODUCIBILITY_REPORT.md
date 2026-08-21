# Reproducibility Check Report

**Project:** Gaussian Process prediction of experimental UV-Vis λ<sub>max</sub> (Beard et al. 2019 corpus)  
**Repository:** https://github.com/drsahanawaz/Bayesian_model  
**Manuscript:** *Gaussian Processes λmax Prediction Exceeds TD-DFT Accuracy* (JCIM target)  
**Report date:** 21 August 2026  
**Environment:** local macOS; conda env `chemo_env`  
**Result:** all manuscript-critical numerical claims reproduced under `random_state = 42`.

---

## 1. Executive summary

Notebooks were executed sequentially (phases 01→08) and key metrics were compared to the manuscript / Supporting Information.  

| Scope | Outcome |
|-------|---------|
| Data curation (n = 6,878 model-ready) | Pass |
| Baseline Ridge / null | Pass |
| Primary fingerprint GP (RMSE / R² / kernel) | Pass (exact) |
| 5-fold CV | Pass (exact) |
| Leakage-free TD-DFT / sTDA holdouts | Pass |
| Hybrid sTDA-feature GP | Pass (exact) |
| Noise floor | Pass (exact) |
| Mechanistic extremes / artifacts | Pass |
| Scaffold split + matched control + QC | Pass (exact) |

Reported manuscript numbers match this check.

Minor code patches were applied for clean automated re-execution (see §6); they do not change reported science for the primary model.

---

## 2. Computational environment

Analyses were run in conda environment **`chemo_env`** (see `environment.yml` / `requirements.txt`).

| Component | Version used in this check |
|-----------|----------------------------|
| Python | 3.10.20 |
| RDKit | 2025.09.5 |
| scikit-learn | 1.7.2 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| SciPy | 1.15.2 |
| Matplotlib | 3.10.9 |
| joblib | 1.5.3 |

**Seeds / split control (as stated in SI §S6.4):**

- All `train_test_split` / scaffold-group splits: `random_state = 42`
- Shuffled `KFold`: `shuffle = True`, `random_state = 42`
- `GaussianProcessRegressor`: `random_state = 42`
- Full GP fits: `n_restarts_optimizer = 2` (CV folds / diagnostic subsamples: `1`, as in SI)

**Source data:** `paper_allDB.csv` from Beard et al. figshare DOI [10.6084/m9.figshare.7619672](https://doi.org/10.6084/m9.figshare.7619672) (not redistributed in the GitHub repo).

---

## 3. Execution protocol

1. Activate `chemo_env`; set writable `MPLCONFIGDIR` / `IPYTHONDIR` under the repo.
2. Working directory = repository root (notebooks folder).
3. Execute with:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace \
     --ExecutePreprocessor.timeout=-1 \
     --ExecutePreprocessor.kernel_name=python3 \
     <notebook>.ipynb
   ```
4. Compare printed metrics and regenerated `scaffold_split_results.json` to manuscript / SI tables.
5. For primary GP, independently reload `gp_full_local_model.joblib` + `scaler_gp.joblib` and recompute test RMSE/R² (cross-check).
6. Hybrid GP metrics re-fit in a dedicated script after `04c` stopped on a late solvent cell (see §5–6).

**Run log directory:** `../repro_run/logs/` (relative to this repo when checked out beside the parent project tree), including per-notebook logs and `hybrid_refit.log`.

---

## 4. Notebook-by-notebook results

| Notebook | Role | Status | Notes |
|----------|------|--------|-------|
| `01_data_load_and_clean.ipynb` | Schema, filters, 254 nm artifact, SMILES | **PASS** | 8,488 → cleaned `beard_uvvis_cleaned.csv` (n = 7,167) |
| `02_featurization.ipynb` | Scoping + features | **PASS** | Model-ready **n = 6,878** |
| `03_baseline_model.ipynb` | Ridge / OLS on 10 descriptors | **PASS** | Train/test **5,502 / 1,376** |
| `04_gaussian_process_model.ipynb` | Descriptor GP diagnosis (degenerate) | **PASS** | Required patches (§6.1); isotropic GP → null RMSE **124.89** (as designed) |
| `04b_solvent_subset_analysis.ipynb` | Solvent ablations | **PASS** | Negative/solvent results consistent with SI narrative |
| `04c_fingerprint_features.ipynb` | Primary GP, CV, QC, Matérn, hybrid | **PASS (science)** / **PARTIAL (full nbconvert)** | Primary/CV/QC/Matérn executed; late solvent `RidgeCV` hit NaN (§5); hybrid re-verified separately |
| `05_tddft_comparison.ipynb` | Session follow-on | **SKIPPED** | Not self-contained; leakage-free QC covered in `04c` |
| `06_noise_floor_estimate.ipynb` | Duplicate noise floor | **PASS** | Pooled σ = **10.07 nm** |
| `07_mechanistic_interpretation.ipynb` | Residuals / extremes | **PASS** | Loads today’s `gp_full_local_model.joblib` |
| `08_scaffold_split_validation.ipynb` | Scaffold split + QC | **PASS** | Regenerated `scaffold_split_results.json` bit-identical to prior reported values |

Approximate wall times (this machine): 01–03 & 06–07 minutes-scale; 04 ~9 min; 04b ~3 min; 04c ~3.2 h before late failure; hybrid refit ~20 min; 08 ~33 min.

---

## 5. Metric verification table

Tolerance: manuscript rounding (e.g. 87.3 vs 87.28) accepted as match when underlying two-decimal printouts agree.

| Claim | Manuscript / SI | Observed this check | Match |
|-------|-----------------|---------------------|-------|
| Curated modeling set | 6,878 | 6,878 | Yes |
| Random split sizes | 5,502 / 1,376 | 5,502 / 1,376 | Yes |
| Tuned Ridge (10 descriptors) | RMSE 121.7 nm; R² 0.050 | 121.67; 0.050 | Yes |
| Null baseline RMSE | 124.9 nm | 124.89 nm | Yes |
| **Primary GP** | **92.15 nm; R² 0.455** | **92.15; 0.455** | **Yes (exact)** |
| Fitted kernel | 169² × RBF(16.9) + White(5.28×10³) | identical string | Yes |
| Calibration r (test) | 0.059 | 0.059 | Yes |
| 5-fold CV | 92.93 ± 1.25; 0.446 ± 0.008 | identical | Yes |
| Matérn full | ~92.17; R² 0.455 | 92.17; 0.455 | Yes |
| TD-DFT holdout (n = 36) | GP 87.3 vs TD-DFT 108.7 | 87.28 vs 108.70 | Yes |
| TD-DFT bias | +57.2 nm | +57.21 nm | Yes |
| sTDA holdout (n = 1,011) | GP 81.6 vs 117.1 | 81.59 vs 117.06 | Yes |
| Hybrid matched baseline | 83.32 nm; R² 0.429 | 83.32; 0.429 | Yes |
| Hybrid GP + sTDA feature | 81.04 nm; R² 0.459 | 81.04; 0.459 | Yes |
| Hybrid coverage n | 5,038 | 5,038 | Yes |
| Noise floor (pooled σ) | 10.07 nm | 10.07 nm | Yes |
| Extreme residuals (cleaned) | +146.5 (n=111); −100.5 (n=107) | identical | Yes |
| Artifact fractions | ~18–22% | 24/135 (17.8%); 30/137 (21.9%) | Yes |
| Scaffold groups / split | 3,643; 4,963 / 1,915; 729 unseen | identical | Yes |
| Scaffold GP | 105.93; R² 0.241 | identical | Yes |
| Matched-size random control | 93.85; R² 0.438 | identical | Yes |
| Scaffold Ridge | 111.30; R² 0.162 | identical | Yes |
| Scaffold calib. r | −0.030 | −0.03 | Yes |
| Scaffold TD-DFT (n = 28) | GP 84.9 vs 121.4; bias +60.4 | 84.92 vs 121.36; +60.39 | Yes |
| Scaffold sTDA (n = 1,477) | GP 99.4 vs 126.0; bias +65.3 | 99.39 vs 125.96; +65.28 | Yes |

### Independent primary-GP reload check

After `04c` wrote `gp_full_local_model.joblib` / `scaler_gp.joblib`, features were rebuilt (256-bit Morgan radius 2 + `LargestConjugatedSystemSize`), split with `random_state=42`, and scored:

```text
PRIMARY GP: RMSE=92.15 nm  R2=0.455  calib_r=0.059
kernel: 169**2 * RBF(length_scale=16.9) + WhiteKernel(noise_level=5.28e+03)
```

Artifact: `_primary_gp_check.json`.

### Regenerated scaffold summary

Artifact: `scaffold_split_results.json` (byte-level agreement with previously archived values for all reported fields).

---

## 6. Issues found and remediation

### 6.1 `04_gaussian_process_model.ipynb`

| Issue | Impact | Fix applied |
|-------|--------|-------------|
| `n_features` used but undefined | `NameError`; notebook not executable | Set `n_features = X_train_scaled.shape[1]` after scaling |
| “Quick” ARD GP on full n ≈ 5,502 | Multi-hour hang; blocks automated re-runs | ARD quick-check fit on **1,500-point subsample** (same subsample used later in the notebook). Scientific intent (ARD vs degenerate isotropic) preserved; full-n ARD is not a manuscript primary metric |

### 6.2 `04c_fingerprint_features.ipynb`

| Issue | Impact | Fix applied |
|-------|--------|-------------|
| `RidgeCV` on fingerprint + ET(30) with NaN ET(30) rows | Under scikit-learn 1.7.x, **all CV fits fail** (`Input X contains NaN`); `nbconvert` aborts | Cells **77** and **81**: drop rows with any NaN before split/fit; print retained row counts |
| Full notebook wall time | ~hours (many full GPs + 5-fold CV) | Expected; use `timeout=-1` |

**Note:** After the NaN failure, hybrid GP cells had not been freshly re-fit in that `nbconvert` process. Hybrid metrics were therefore **re-fit in a dedicated script** matching notebook logic (`random_state=42`, n = 5,038) and matched SI Table S6 exactly (81.04 / 83.32). Models rewritten: `gp_stda_model.joblib`, `scaler_gp_stda.joblib`, `gp_base_matched_model.joblib`.

### 6.3 `05_tddft_comparison.ipynb`

Depends on in-memory objects from a live `04c` session (`fp_df`, `gp_full_local`, `scaler_gp`). Not suitable for standalone `nbconvert`. Leakage-free TD-DFT/sTDA numbers are produced inside **`04c`** and were verified there.

---

## 7. What was *not* claimed / caveats

1. **Bit-for-bit identity across OS/BLAS builds** is not guaranteed; this check used one macOS ARM host with pinned conda packages.
2. **Notebook 04c:** a late solvent-Ridge `NaN` edge case was patched; primary hybrid metrics were verified independently (see hybrid check above). Reported hybrid metrics do not depend on completing a second full `nbconvert` pass of the patched notebook.
3. **Large `.joblib` files** remain gitignored; third parties can regenerate them from the notebooks, or request them from the corresponding author.
4. **Raw `paper_allDB.csv`** must be obtained from figshare; curated intermediates can be regenerated or shipped separately per Data and Software Availability statement.

---

## 8. Artifacts produced by this check

| Artifact | Purpose |
|----------|---------|
| `scaffold_split_results.json` | Scaffold / control / QC numeric summary |
| `gp_full_local_model.joblib`, `scaler_gp.joblib` | Primary GP (regenerated) |
| `gp_stda_model.joblib`, `scaler_gp_stda.joblib`, `gp_base_matched_model.joblib` | Hybrid pair (regenerated) |
| `gp_scaffold_model.joblib`, `gp_matched_control_model.joblib` | Scaffold models (regenerated by 08) |
| `_primary_gp_check.json` | Independent primary reload scores |
| `_hybrid_check.json` | Hybrid refit scores |
| `_repro_final_report.json` | Machine-readable pass table |
| This file (`REPRODUCIBILITY_REPORT.md`) | Human-readable report |

---

## 9. Conclusion

Under the documented environment and **`random_state = 42`**, the analysis pipeline reproduces the manuscript’s central quantitative claims, including primary GP performance, leakage-free quantum-chemical comparisons, hybrid sTDA ablation, noise floor, residual extremity statistics, and Bemis–Murcko scaffold-split results.

Reported manuscript metrics are consistent with this verification.

---

*Reproducibility verification aligned with ACS Research Data Policy Level 2 / JCIM data and software availability expectations.*
