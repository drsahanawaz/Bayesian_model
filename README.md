# Gaussian Process Prediction of UV-Vis λ<sub>max</sub>

Analysis code accompanying the manuscript:

> **Gaussian Process λ<sub>max</sub> Prediction Outperforms TD-DFT Benchmarks**  
> Md Sahanawaz — Sheikhpara ARM Polytechnic  
> Intended venue: *Journal of Chemical Information and Modeling* (ACS)

This repository provides Jupyter notebooks that curate the Beard et al. (2019) literature-mined UV-Vis dataset, train a Gaussian Process (GP) on 256-bit Morgan fingerprints plus largest conjugated-system size, and benchmark against TD-DFT / sTDA under leakage-free and Bemis–Murcko scaffold-split evaluation.

| | |
|---|---|
| **Code** | https://github.com/drsahanawaz/Bayesian_model |
| **License** | [MIT](LICENSE) |
| **Source data** | Beard et al., figshare [10.6084/m9.figshare.7619672](https://doi.org/10.6084/m9.figshare.7619672) |
| **Random seed** | `random_state = 42` (all splits and shuffled CV) |
| **Reproducibility report** | [REPRODUCIBILITY_REPORT.md](REPRODUCIBILITY_REPORT.md) |

---

## Key results (primary model)

| Setting | Metric | Value |
|---|---|---|
| Random split (n<sub>train</sub> = 5,502) | RMSE / R² | 92.15 nm / 0.455 |
| Scaffold split (n<sub>train</sub> = 4,963) | RMSE / R² | 105.93 nm / 0.241 |
| Leakage-free TD-DFT holdout (n = 36) | GP vs TD-DFT RMSE | 87.3 vs 108.7 nm |
| Leakage-free sTDA holdout (n = 1,011) | GP vs sTDA RMSE | 81.6 vs 117.1 nm |

Numeric scaffold-split summary: [`scaffold_split_results.json`](scaffold_split_results.json). Example figures are included as PNGs.

---

## Data and software availability (JCIM-aligned)

Consistent with JCIM requirements on method/data sharing ([Merz et al., *J. Chem. Inf. Model.* **2020**, *60*, 5868–5869](https://doi.org/10.1021/acs.jcim.0c01389)) and ACS Research Data Policy Level 2:

| Material | Availability |
|---|---|
| Raw Beard UV-Vis database (`paper_allDB.csv`) | Download from figshare DOI above (not redistributed here) |
| Curated intermediates | Tracked in this repo: `beard_uvvis_cleaned.csv`, `beard_model_ready_features.csv`, `beard_baseline_features.csv`, `scaffold_split_assignments.csv`, `solvent_et30_data.csv` |
| Analysis notebooks (phases 01–08) | This repository (MIT License) |
| Software environment | [`environment.yml`](environment.yml) / [`requirements.txt`](requirements.txt) with pinned versions |
| Trained `.joblib` GP models | Not hosted (file size); regenerate with notebooks `04c` and `08` |

**Archival DOI.** Upon acceptance, a versioned GitHub Release will be archived on [Zenodo](https://zenodo.org/) to provide a permanent DOI in addition to this GitHub URL.

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
| `scaffold_split_results.json` | Phase 8 numeric summary |
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
3. Create the conda environment above.
4. Run notebooks **in order** from the repository root:

```bash
conda activate chemo_env
cd /path/to/Bayesian_model

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 \
  01_data_load_and_clean.ipynb
# then 02 → 03 → 04 → 04b → 04c → 06 → 07 → 08
```

### Notes

- All `train_test_split` / scaffold-group splits and shuffled `KFold` folds use **`random_state = 42`**.
- GP fits use `n_restarts_optimizer = 2` for full training runs (`= 1` for CV folds / diagnostic subsamples), as in the Supporting Information.
- `04c` and `08` are computationally heavy (full GPs on thousands of compounds).
- `07` expects `gp_full_local_model.joblib` and `scaler_gp.joblib` written by `04c`.
- Leakage-free TD-DFT / sTDA manuscript numbers are generated inside **`04c`** (preferred over standalone `05`).

---

## Citation

Please cite the accompanying JCIM manuscript when available, and the source dataset:

```text
Beard, E. J.; Sivaraman, G.; Vázquez-Mayagoitia, Á.; Vishwanath, V.; Cole, J. M.
Comparative dataset of experimental and computational attributes of UV/vis absorption spectra.
Sci. Data 2019, 6, 307. https://doi.org/10.1038/s41597-019-0306-0
```

```text
Beard et al. UV/vis dataset (figshare). https://doi.org/10.6084/m9.figshare.7619672
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
