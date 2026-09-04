# Gaussian Process Prediction of UV-Vis λ<sub>max</sub>

Analysis code accompanying the manuscript:

> **Benchmark Label Definition and Same-Corpus Evaluation Explain Reported Machine-Learning Advantages over TD-DFT in λ<sub>max</sub> Prediction**  
> Md Sahanawaz — Sheikhpara ARM Polytechnic

This repository provides Jupyter notebooks that curate the Beard et al. (2019) literature-mined UV-Vis dataset, train a Gaussian Process (GP) on 256-bit Morgan fingerprints plus largest conjugated-system size, and benchmark against TD-DFT / sTDA under leakage-free and Bemis–Murcko scaffold-split evaluation.

Notebooks 09–13 apply the same published model, without retraining, to two independent external corpora that differ in label convention, and score both against published quantum-chemical calculations.

**Label definition governs calibration.** On 318 photoswitches labelled with one specified electronic transition, the GP shows a +53.5 nm systematic red shift and negative R². On 5,811 compounds from a second literature compilation whose label convention matches training, it is near-unbiased and transfers about as well as it does to its own novel scaffolds, despite a larger fraction of unseen scaffolds. Chemical novelty degrades performance gradually; changing the label definition breaks it.

**The advantage over quantum chemistry does not survive external validation.** The GP outperforms TD-DFT and sTDA only on holdouts drawn from its training corpus. Against published ωB97X-D3 vertical excitations on 5,501 training-disjoint compounds from the mixed-convention corpus it is outperformed (RMSE 95.0 vs 84.1 nm; Pearson r 0.46 vs 0.85); after removing each method's systematic offset the gap is 95 vs 57 nm. A six-part fairness audit (notebook 13) does not reverse that ordering. Reported margins over quantum chemistry are properties of the evaluation design, not of the method alone.

| | |
|---|---|
| **Code** | https://github.com/drsahanawaz/Bayesian_model |
| **License** | [MIT](LICENSE) |
| **Source data** | Beard et al., figshare [10.6084/m9.figshare.7619672](https://doi.org/10.6084/m9.figshare.7619672) |
| **Second external corpus** | Joung et al., figshare [10.6084/m9.figshare.12045567](https://doi.org/10.6084/m9.figshare.12045567) (CC BY 4.0, redistributed here) |
| **Computed reference** | Greenman et al., *Chem. Sci.* **2022**, *13*, 1152 — ωB97X-D3 vertical excitations ([uvvisml](https://github.com/learningmatter-mit/uvvisml), MIT, redistributed here) |
| **Random seed** | `random_state = 42` (all splits and shuffled CV) |
| **Reproducibility report** | [REPRODUCIBILITY_REPORT.md](REPRODUCIBILITY_REPORT.md) |

---

## Key results

**Within the training corpus** — the GP outperforms both quantum-chemical references, and that holds under scaffold splitting. Note that both quantum-chemical comparisons below are drawn from the corpus that supplied the training labels, and neither reproduces externally (see the external quantum-chemistry table further down):

| Setting | Metric | Value |
|---|---|---|
| Random split (n<sub>train</sub> = 5,502) | RMSE / R² | 92.15 nm / 0.455 |
| Scaffold split (n<sub>train</sub> = 4,963) | RMSE / R² | 105.93 nm / 0.241 |
| Leakage-free TD-DFT holdout (n = 36) | GP vs TD-DFT RMSE | 87.3 vs 108.7 nm |
| Leakage-free sTDA holdout (n = 1,011) | GP vs sTDA RMSE | 81.6 vs 117.1 nm |

**On the external photoswitch holdout** — the same model applied without retraining, against TD-DFT values computed for the transition the labels denote:

| Setting | Metric | Value |
|---|---|---|
| External holdout (n = 318) | RMSE / R² / bias | 77.02 nm / −0.269 / +53.45 nm |
| CAM-B3LYP/6-31G\*\* subset (n = 114) | DFT vs GP RMSE | 25.40 vs 80.90 nm |
| PBE0 subset (n = 111) | DFT vs GP RMSE | 33.42 vs 75.99 nm |
| Curated ε ≥ 10k vs size-matched control | External RMSE | 62.11 vs 79.39 nm |
| Azo-class-matched vs size-matched control | External RMSE | 87.20 vs 73.72 nm |

**On the second external corpus** — chemically novel, but label convention compatible with training:

| Setting | Metric | Value |
|---|---|---|
| Joung holdout, all (n = 5,811) | RMSE / R² / bias | 94.75 nm / +0.209 / −8.79 nm |
| Unseen-scaffold subset (n = 4,864) | RMSE / R² / bias | 98.68 nm / +0.171 / −15.71 nm |
| Scaffolds absent from training | Fraction | 83.7% (vs 38.1% for the scaffold split) |
| Overlap with training removed | Structures | 294 (4.8%); 1 residual at skeleton level |

**Against published quantum chemistry on the second external corpus** — the same experimental labels as above, scored against ωB97X-D3/def2-SVPD vertical excitations that this work did not compute (n = 5,501 matched, training-disjoint, null RMSE 106.4 nm):

| Reference | RMSE | Offset | Scatter | Affine RMSE | R² | r |
|---|---|---|---|---|---|---|
| ωB97X-D3, lowest excitation S₁ | 84.1 nm | −61.9 nm | 57.0 nm | 58.2 nm | +0.375 | 0.850 |
| ωB97X-D3, brightest of five | 110.2 nm | −87.4 nm | 67.2 nm | 68.2 nm | −0.074 | 0.775 |
| Published GP | 95.0 nm | −8.5 nm | 94.6 nm | 94.6 nm | +0.203 | 0.459 |

Paired bootstrap over compounds: the calculations hold an 11.0 nm RMSE advantage (95% CI 4.3–15.5) and a 0.395 advantage in Pearson r (95% CI 0.319–0.450). The two methods fail in opposite ways — the calculations carry a large removable offset and modest scatter, the GP almost no offset and a scatter that is 89% of the null. On the 4,584 unseen-scaffold compounds the gap widens to 87.1 vs 99.0 nm. Six fairness threats were tested (coverage, structure matching, vacuum-vs-solution, five-state truncation, correctability, applicability domain) and none reverses the ordering; the vacuum and vertical-excitation approximations work *against* the calculations, so correcting them would widen the gap.

**The four-tier audit.** R² falls from +0.455 (random split) to +0.241 (scaffold split) to +0.171 (second corpus, novel scaffolds), then inverts to −0.269 (photoswitches). Signed bias stays within ±16 nm across the first three and jumps to +53.5 nm on the fourth. Bootstrap 95% intervals for the two external corpora do not overlap on either metric. Chemical novelty degrades performance gradually; reassigning the label breaks it.

**Uncertainty.** GP predictive uncertainty is too weak to triage predictions in any regime tested. The raw correlation between predicted σ and absolute error is |r| ≤ 0.06 within the training corpus and −0.019 on the photoswitch holdout. It reaches +0.373 on the second corpus, but this tracks that corpus's wider label distribution rather than chemical familiarity: matched to the photoswitch holdout's λmax distribution it falls to +0.155 ± 0.061, and a dose–response sweep shows r increasing with label spread alone from 0.107 to 0.206. Calibration correlations are therefore not comparable across evaluation sets of differing label spread.

Numeric summaries: [`scaffold_split_results.json`](scaffold_split_results.json), [`results_external_dft_benchmark.json`](results_external_dft_benchmark.json), [`results_label_curation_transfer.json`](results_label_curation_transfer.json), [`results_second_external_corpus.json`](results_second_external_corpus.json), [`results_external_tddft_benchmark.json`](results_external_tddft_benchmark.json). Example figures are included as PNGs.

---

## Data and software availability

| Material | Availability |
|---|---|
| Raw Beard UV-Vis database (`paper_allDB.csv`) | Download from figshare DOI above (not redistributed here) |
| Photoswitch compilation (`photoswitches.csv`, `purchasable_switch.csv`) | Obtain from Griffiths et al., *Chem. Sci.* **2022**, *13*, 13541 (not redistributed here pending licence review) |
| Joung chromophore database (`joung_chromophore_db.csv`) | **Redistributed in this repo** under CC BY 4.0; source figshare [10.6084/m9.figshare.12045567](https://doi.org/10.6084/m9.figshare.12045567) v3. Notebook 12 therefore runs from a clean clone with no external download |
| Computed ωB97X-D3 excitations (`uvvisml_computed_tddft.csv`) | **Redistributed in this repo** under the MIT licence of [learningmatter-mit/uvvisml](https://github.com/learningmatter-mit/uvvisml), with attribution to Greenman et al., *Chem. Sci.* **2022**, *13*, 1152 (archived at [10.5281/zenodo.5773155](https://doi.org/10.5281/zenodo.5773155)). Notebook 13 runs from a clean clone; it re-fetches the file from source if absent. **No excited-state calculations were performed in this work.** |
| Curated intermediates | Tracked in this repo: `beard_uvvis_cleaned.csv`, `beard_model_ready_features.csv`, `beard_baseline_features.csv`, `scaffold_split_assignments.csv`, `solvent_et30_data.csv` |
| External holdouts and predictions | Tracked in this repo: `external_validation.csv`, `external_gp_vs_dft_predictions.csv`, `external_validation_joung.csv`, `external_gp_vs_tddft_joung.csv` |
| Analysis notebooks (phases 01–13) | This repository (MIT License) |
| Software environment | [`environment.yml`](environment.yml) / [`requirements.txt`](requirements.txt) with pinned versions |
| Trained `.joblib` GP models | Not hosted (file size); regenerate with notebooks `04c` and `08` |

**Archival DOI.** A versioned GitHub Release will be archived on [Zenodo](https://zenodo.org/) to provide a permanent DOI in addition to this GitHub URL. The DOI will be added here once minted.

---

## Repository contents

| File | Description |
|---|---|
| `01_data_load_and_clean.ipynb` | Schema repair, missingness, range filter, 254 nm artifact, SMILES validation, duplicates |
| `02_featurization.ipynb` | Domain scoping, RDKit descriptors, mechanism features, collinearity |
| `03_baseline_model.ipynb` | Ridge / OLS baseline on 10 hand-selected descriptors |
| `04_gaussian_process_model.ipynb` | Initial GP on descriptors (degenerate / weak solution) |
| `04b_solvent_subset_analysis.ipynb` | Solvent curation and ablations |
| `04c_fingerprint_features.ipynb` | Morgan fingerprints, ablations, **primary GP**, leakage-free QC, hybrid sTDA |
| `05_tddft_comparison.ipynb` | TD-DFT holdout diagnostics (session notebook; prefer `04c` for standalone QC) |
| `06_noise_floor_estimate.ipynb` | Duplicate-based experimental noise floor |
| `07_mechanistic_interpretation.ipynb` | Residual / extremity analysis (loads models from `04c`) |
| `08_scaffold_split_validation.ipynb` | Bemis–Murcko split, matched-size control, scaffold-consistent QC |
| `09_external_validation.ipynb` | External holdout construction (stereo-free InChIKey matching); published GP applied unchanged |
| `10_external_dft_benchmark.ipynb` | Coverage-matched GP vs four levels of theory; uncertainty binning |
| `11_label_curation_transfer.ipynb` | Curated training subsets vs size-matched random controls; affine recalibration floor |
| `12_second_external_corpus.ipynb` | Second external holdout (Joung); scaffold-novelty analysis, four-tier comparison, uncertainty range-dependence controls |
| `13_external_tddft_benchmark.ipynb` | Published ωB97X-D3 excitations matched to the Joung holdout; four transition-selection conventions, error decomposition, cross-validated affine correction, six-part fairness audit |
| `_build_nb10.py` … `_build_nb13.py` | Deterministic builders that regenerate notebooks 10–13 |
| `make_preprint_figures.py` | Composite manuscript Figures 5, 6, 7 and 8 |
| `scaffold_split_results.json` | Phase 8 numeric summary |
| `results_external_dft_benchmark.json` | Phase 10 numeric summary |
| `results_label_curation_transfer.json` | Phase 11 numeric summary |
| `results_second_external_corpus.json` | Phase 12 numeric summary |
| `results_external_tddft_benchmark.json` | Phase 13 numeric summary, including the full fairness audit |
| `environment.yml` / `requirements.txt` | Pinned software environment |
| `LICENSE` | MIT License |

---

## Environment (pinned)

Analyses reported in the manuscript were developed with:

| Package | Version | Role |
|---|---|---|
| Python | 3.10.20 | Runtime |
| RDKit | 2025.09.5 | Parsing, fingerprints, scaffolds, descriptors |
| scikit-learn | 1.7.2 | GP, Ridge/Lasso, scalers, splits, CV |
| NumPy | 2.2.6 | Arrays |
| pandas | 2.3.3 | Tables |
| SciPy | 1.15.2 | Numerics |
| Matplotlib | 3.10.9 | Figures |
| seaborn | 0.13.2 | Figures |
| joblib | 1.5.3 | Model I/O |

```bash
conda env create -f environment.yml
conda activate chemo_env
```

Exact bit-for-bit identity can still vary slightly across BLAS/OS builds; use this environment for closest reproduction.

---

## How to reproduce

1. Clone this repository.
2. Download `paper_allDB.csv` from the Beard figshare record and place it in the repository root.
3. For notebooks 09–11 only, obtain the photoswitch compilation from Griffiths et al. and place `photoswitches.csv` in the repository root and `purchasable_switch.csv` in its **parent** directory. Neither third-party file is redistributed here. Notebooks 12 and 13 need no download: their source CSVs are CC BY 4.0 and MIT respectively and ship with the repository.
4. Create the conda environment above.
5. Run notebooks **in order** from the repository root:

```bash
conda activate chemo_env
cd /path/to/Bayesian_model

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  01_data_load_and_clean.ipynb
# then 02 → 03 → 04 → 04b → 04c → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13
python make_preprint_figures.py
```

### Notes

- All `train_test_split` / scaffold-group splits and shuffled `KFold` folds use **`random_state = 42`**.
- GP fits use `n_restarts_optimizer = 2` for full training runs (`= 1` for CV folds / diagnostic subsamples), as in the Supporting Information.
- `04c` and `08` are computationally heavy (full GPs on thousands of compounds).
- `07` and `09`–`12` expect `gp_full_local_model.joblib` and `scaler_gp.joblib` written by `04c`. `13` does not: it reuses the predictions written by `12`, so it runs in under a minute without loading a model.
- Leakage-free TD-DFT / sTDA manuscript numbers are generated inside **`04c`** (preferred over standalone `05`).
- `09`–`12` apply the published GP without refitting; the training-fold scaler is used in `transform` mode only, so holdout feature moments never leak.
- Size-matched control draws in `11` use seeds 0–2; everything else uses `random_state = 42`.
- Notebooks `10`–`13` are generated deterministically by `_build_nb10.py` … `_build_nb13.py`, and reproduce identical values on repeat execution.
- `13` performs **no excited-state calculations**. It consumes published ωB97X-D3 values and its only stochastic step is a 2,000-resample paired bootstrap under `default_rng(42)`.
- `12` uses bootstrap and resampling throughout (2,000 resamples for confidence intervals, 200–400 draws for the calibration controls) under a single `default_rng(42)` stream, so its intervals are reproducible but will shift slightly if cells are executed out of order.

---

## Citation

Please cite the accompanying manuscript when available, and all four source datasets:

```text
Beard, E. J.; Sivaraman, G.; Vázquez-Mayagoitia, Á.; Vishwanath, V.; Cole, J. M.
Comparative dataset of experimental and computational attributes of UV/vis absorption spectra.
Sci. Data 2019, 6, 307. https://doi.org/10.1038/s41597-019-0306-0
```

```text
Beard et al. UV/vis dataset (figshare). https://doi.org/10.6084/m9.figshare.7619672
```

```text
Griffiths, R.-R.; Greenfield, J. L.; Thawani, A. R.; Jamasb, A. R.; Moss, H. B.; Bourached, A.;
Jones, P.; McCorkindale, W.; Aldrick, A. A.; Fuchter, M. J.; Lee, A. A.
Data-driven discovery of molecular photoswitches with multioutput Gaussian processes.
Chem. Sci. 2022, 13, 13541-13551.
```

```text
Joung, J. F.; Han, M.; Jeong, M.; Park, S.
Experimental database of optical properties of organic compounds.
Sci. Data 2020, 7, 295. https://doi.org/10.1038/s41597-020-00634-8
Dataset: DB for chromophore, v3. figshare. https://doi.org/10.6084/m9.figshare.12045567 (CC BY 4.0)
```

```text
Greenman, K. P.; Green, W. H.; Gomez-Bombarelli, R.
Multi-fidelity prediction of molecular optical peaks with deep learning.
Chem. Sci. 2022, 13, 1152-1162. https://doi.org/10.1039/D1SC05677H
Computed data: https://github.com/learningmatter-mit/uvvisml (MIT);
archived at https://doi.org/10.5281/zenodo.5773155
```

---

## Author / contact

**Md Sahanawaz**  
Department of Science & Humanities, Sheikhpara ARM Polytechnic, West Bengal, India  
Email: sahanawaz@wbscte.ac.in

Independent, unfunded research. No competing financial interest.

---

## License

Analysis code and curated workflow files in this repository are released under the [MIT License](LICENSE).  
The Beard source database remains subject to its original figshare / *Scientific Data* terms and is not redistributed here.  
`joung_chromophore_db.csv` is redistributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and remains attributable to Joung et al.; it is unmodified from figshare v3.  
`uvvisml_computed_tddft.csv` is redistributed under the [MIT License](https://github.com/learningmatter-mit/uvvisml/blob/main/LICENSE) of the `uvvisml` repository and remains attributable to Greenman et al.; it is unmodified from that repository's `20210109_computed_df_all.csv`. No excited-state calculations were performed in this work.
