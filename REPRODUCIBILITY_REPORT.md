# Reproducibility Check Report

**Project:** Gaussian Process prediction of experimental UV-Vis λ<sub>max</sub> (Beard et al. 2019 corpus)  
**Repository:** https://github.com/drsahanawaz/Bayesian_model  
**Manuscript:** *Benchmark Label Definition and Same-Corpus Evaluation Explain Reported Machine-Learning Advantages over TD-DFT in λmax Prediction*  
**Report date:** 21 August 2026; notebooks 09–11 checked 28 August 2026; notebooks 12–13 checked 31 August 2026  
**Environment:** local macOS; conda env `chemo_env`  
**Result:** all manuscript-critical numerical claims reproduced under `random_state = 42`.

---

## 1. Executive summary

Notebooks were executed sequentially (01→13) and key metrics were compared to the manuscript / Supporting Information.  

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
| External holdout construction and GP transfer | Pass (exact) |
| Coverage-matched TD-DFT comparison | Pass (exact) |
| Label-curation experiments and recalibration floor | Pass (exact) |
| Second external corpus: curation, GP transfer, four-tier audit | Pass (exact) |
| Uncertainty range-dependence controls | Pass (exact) |
| External ωB97X-D3 benchmark and fairness audit | Pass (exact) |

Reported manuscript numbers match this check.

**Uncertainty wording.** An earlier draft claimed GP predictive uncertainty was "uninformative in all three evaluation regimes (|r| ≤ 0.06)". On the fourth regime (notebook 12) the raw correlation is r = +0.373, but that tracks the evaluation set's label spread rather than chemical familiarity; the manuscript states the scoped claim (see §7).

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
| `09_external_validation.ipynb` | External holdout construction; published GP applied unchanged | **PASS** | 318-compound holdout; RMSE 77.02 nm, R² −0.269, bias +53.45 nm |
| `10_external_dft_benchmark.ipynb` | Coverage-matched GP vs TD-DFT | **PASS** | CAM-B3LYP 25.40 vs GP 80.90 (n = 114); PBE0 33.42 vs GP 75.99 (n = 111) |
| `11_label_curation_transfer.ipynb` | Curated subsets vs size-matched controls; recalibration floor | **PASS** | Bright ε ≥ 10k 62.11 vs control 79.39; azo 87.20 vs control 73.72 |
| `12_second_external_corpus.ipynb` | Second external holdout; four-tier audit; calibration controls | **PASS** | 5,811-compound holdout; RMSE 94.75 nm, R² +0.209, bias −8.79 nm; novel-scaffold subset (n = 4,864) R² +0.171 |
| `13_external_tddft_benchmark.ipynb` | Published ωB97X-D3 excitations vs GP on the mixed-convention holdout; fairness audit | **PASS** | 5,501 matched, training-disjoint; TD-DFT S₁ 84.13 nm / R² +0.375 / r 0.850 vs GP 94.97 nm / +0.203 / 0.459 |

Approximate wall times (this machine): 01–03 & 06–07 minutes-scale; 04 ~9 min; 04b ~3 min; 04c ~3.2 h before late failure; hybrid refit ~20 min; 08 ~33 min; 09 seconds-scale; 10 ~20 s; 11 ~70 s; 12 ~35 s.

**Phase 2 re-execution check (28 August 2026).** Notebooks 10 and 11 were re-executed after consolidating the analysis into this repository layout. All reported values were unchanged. Notebooks 10 and 11 are generated deterministically by `_build_nb10.py` and `_build_nb11.py`.

**Phase 12 check (31 August 2026).** Notebook 12 was built by `_build_nb12.py` and executed three times end to end during development; every reported metric was identical across runs. Independent verification points:

- The source CSV was downloaded fresh from figshare and its licence confirmed as CC BY 4.0 through the figshare API before use.
- Overlap removal was validated at two granularities. The stereochemistry-free InChIKey used for removal eliminated 294 structures; an independent recheck on the InChIKey skeleton block, which collapses tautomers, salts and protonation states, found one residual match. Excluding it moved RMSE from 94.748 to 94.743 nm and left R² at +0.209, so no reported result depends on contamination handling.
- The four-tier comparison quotes tiers 1 and 2 from notebooks 04c and 08 rather than recomputing them; tiers 3 and 4 are computed in place with 2,000-resample bootstrap intervals. The R² intervals for the two external corpora, [+0.144, +0.195] and [−0.589, −0.034], do not overlap.
- The uncertainty finding was tested against three competing explanations rather than asserted. An earlier contiguous-window variance match was discarded during development because it confounded label spread with chemistry (≈34 versus ≈273 scaffolds per draw); the binned resampling retained in the notebook preserves chemical diversity and is the version reported.

`make_preprint_figures.py` reads the tracked `results_label_curation_transfer.json` (not a gitignored `_*.json` alias), so figure generation succeeds on a clean clone.

**Phase 13 check (31 August 2026).** Notebook 13 was built by `_build_nb13.py`. Its central result reverses the in-corpus GP-versus-TD-DFT ordering, so the fairness of that comparison was audited in full before the result was accepted.

The finding: on the external mixed-convention holdout, published ωB97X-D3/def2-SVPD vertical excitations reach RMSE 84.13 nm and Pearson r 0.850 where the published GP reaches 94.97 nm and 0.459. This contradicts the in-corpus result of §3.2, in which the GP outperforms TD-DFT 87.3 to 108.7 nm.

Six threats to the comparison's fairness were tested, and the finding was accepted only after all six failed to overturn it:

- **Coverage.** 5,501 of 5,811 holdout compounds matched (94.7%). The 310 unmatched compounds are larger (mean MW 529.5 vs 468.9) and more conjugated (29.3 vs 26.3), which would ordinarily suggest the matched subset was enriched in easy molecules — but the GP predicts the unmatched compounds *better* (RMSE 90.76 vs 94.97 nm), so no such enrichment exists.
- **Structure matching.** Isotope labels were cleared before matching, collapsing 78 of the 5,501 matched compounds from isotopologue groups. Excluding all 78 moves TD-DFT from 84.13 to 83.89 nm and the GP from 94.97 to 95.13 nm.
- **Vacuum vs solution.** Single-solvent compounds (n = 3,432) give RMSE 83.39 nm and scatter 59.35 nm; multi-solvent compounds (n = 2,069) give 85.33 and 52.46 nm. Comparable, and the within-structure solvent spread measured in phase 12 is 6.6 nm, an order of magnitude below the scatter at issue.
- **State truncation.** 94.5% of experimental maxima lie redder than the computed S₁ and 0.0% above S₅, so the −61.9 nm offset is a systematic overestimation of excitation energy rather than an artifact of the five-state window.
- **Correctability.** An affine correction was fitted by five-fold cross-validation so that neither method is scored on molecules used to fit its own correction. TD-DFT improves from 84.13 to 58.22 nm; the GP is unchanged at 94.56 nm, having no correctable component. A 2,000-resample paired bootstrap over compounds gives TD-DFT an 11.0 nm RMSE advantage (95% CI 4.3–15.5) and a 0.395 Pearson-r advantage (95% CI 0.319–0.450); both intervals exclude zero.
- **Applicability domain.** On the 4,584 unseen-scaffold compounds the ordering widens (87.07 vs 98.95 nm) rather than narrowing.

Data hygiene notes:

- 45 of 28,803 published calculations report excitation energies of 0 eV or of tens of thousands of eV, which are failed calculations. A physicality screen (all five states within 0.3–12 eV) removes them. **None fell in the matched set**, so every reported metric is identical with and without the screen; it is retained to prevent infinities propagating into aggregates over the full table.
- The two approximations in the reference — vacuum, and vertical excitations without vibronic correction — both act *against* the calculations. Correcting them would widen the reported gap, so the conclusion is robust in the conservative direction.
- Notebook 13 loads no `.joblib` model. It reuses the predictions written by notebook 12, so it is fast and cannot re-introduce a scaler or refitting error.

**Scope of the GP-versus-TD-DFT claim.** The manuscript states that the GP outperforms TD-DFT and sTDA only on holdouts drawn from the training corpus. The external ωB97X-D3 comparison (notebook 13) does not reproduce an out-of-corpus advantage; that limitation is intentional and is reflected in the manuscript title and abstract.

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
5. **Photoswitch data are not redistributed** pending licence review, so notebooks 09–11 need `photoswitches.csv` supplied by the user. Notebook 12 is exempt: its source CSV is CC BY 4.0 and ships with the repository.
6. **Label quality and label convention are not separated** by the second external corpus. Joung labels are manually curated to a stated convention (first absorption maximum) while Beard labels are automatically extracted, so the tier-3 result confounds cleaner labels with compatible labels; both would act in the same direction. What the comparison does establish is that a chemically unfamiliar corpus with a compatible convention is predicted acceptably while a chemically narrow corpus with an incompatible one is not.
7. **Tier-3 transfer is comparative, not absolute.** On the second corpus the model improves on the null baseline by only 11% in RMSE (94.75 vs 106.56 nm), Pearson r is 0.466, and predictions are visibly regressed to the mean. The supported claim is that this corpus is no harder than the model's own novel scaffolds, not that the model predicts it well.
8. **Predictive uncertainty is weak, not absent.** At matched label distribution the correlation between predicted σ and absolute error is +0.155 with a 95% range of +0.035 to +0.285, so the lower bound excludes zero. The operational conclusion — too weak to triage predictions — still holds; "uninformative" would overstate the result.
9. **The GP's advantage over quantum chemistry is not claimed outside the training corpus.** It holds on the leakage-free in-corpus holdouts (87.3 vs 108.7 nm against TD-DFT, n = 36; 81.6 vs 117.1 nm against sTDA, n = 1,011) and on neither external corpus. Phase 13 records the inversion and the audit supporting it. Any statement that the model is competitive with excited-state calculation off its own corpus is unsupported by this repository.
10. **The external quantum-chemical reference is a single level of theory.** ωB97X-D3/def2-SVPD vertical excitations in vacuum, taken as published. Its absolute offset is not a measure of what careful excited-state work could achieve on these molecules. The direction of that limitation favours the GP, so it does not threaten the comparison, but the offset should not be quoted as a functional benchmark.
11. **The transition-selection sweep in phase 13 is a weaker demonstration than it appears.** Because 94.5% of experimental maxima lie redder than the computed S₁, S₁ is the reddest available state and every other convention is arithmetically forced to be worse. The sweep is reported with that caveat stated; the phase-10 photoswitch comparison, where the calculations were nearly unbiased, is the cleaner test of label convention.

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
| `results_second_external_corpus.json` | Phase 12 numeric summary: curation funnel, four-tier table, calibration controls |
| `external_validation_joung.csv` | Second external holdout with predictions, σ and residuals |
| `Figure6_second_external_corpus.png` | Manuscript **Figure 6** — four-tier generalisation audit |
| `results_external_tddft_benchmark.json` | Phase 13 numeric summary: reference provenance, matching, all six audit sections, scoreboard, paired bootstrap |
| `external_gp_vs_tddft_joung.csv` | Matched holdout with the five computed excitations, oscillator strengths, all four convention wavelengths and GP predictions |
| `uvvisml_computed_tddft.csv` | Published ωB97X-D3 reference, redistributed under MIT with attribution |
| `Figure7_external_tddft_benchmark.png` | Manuscript **Figure 7** — external quantum-chemical benchmark |
| This file (`REPRODUCIBILITY_REPORT.md`) | Human-readable report |

---

## 9. Conclusion

Under the documented environment and **`random_state = 42`**, the analysis pipeline reproduces the manuscript’s central quantitative claims, including primary GP performance, leakage-free quantum-chemical comparisons, hybrid sTDA ablation, noise floor, residual extremity statistics, Bemis–Murcko scaffold-split results, both external holdouts, the four-tier generalisation audit, and the external quantum-chemical benchmark of phase 13.

Reported manuscript metrics are consistent with this verification. Phase 13 confirms that the GP-versus-TD-DFT advantage does not extend to the external mixed-convention corpus; manuscript claims are scoped accordingly.

---

*Reproducibility verification: every numerical claim in the manuscript and Supporting Information is regenerable from the notebooks in this repository under the pinned environment.*
