"""Build 12_second_external_corpus.ipynb.

Documents the second external validation. The photoswitch holdout of notebooks 09
and 10 changes two things at once relative to training: the chemistry and the
label convention. This notebook adds a corpus that changes only the chemistry, so
the two causes can be separated. It also tests whether the apparently better
uncertainty calibration on that corpus is a genuine signal or an artifact of its
wider label distribution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

MD: Final[str] = "markdown"
CODE: Final[str] = "code"

CELLS: list[tuple[str, str]] = []


def add(kind: str, body: str) -> None:
    """Append a cell, trimming surrounding blank lines."""
    CELLS.append((kind, body.strip("\n")))


# ---------------------------------------------------------------- 0
add(
    MD,
    r"""
**Phase 12 — A second external corpus: separating chemical novelty from label semantics**

Notebooks `09`–`11` established that the published Gaussian Process (GP) fails on the
318-molecule photoswitch holdout, with $R^{2}=-0.269$ and a systematic $+53.5$ nm red
shift, and that CAM-B3LYP/6-31G\*\* is roughly twice as accurate on coverage-matched
molecules even after the GP is granted its entire offset.

Notebook `11` argued from *internal* evidence that the cause is label semantics rather
than chemical novelty: curating training labels by transition character helped, while
matching the holdout's chemical class hurt. That argument is indirect, because the
photoswitch holdout changes **two** things at once relative to training:

1. the **chemistry** — every compound carries an azo linkage, against 6.3% of training;
2. the **label convention** — the label is specifically the E-isomer
   $\pi\rightarrow\pi^{\ast}$ band, rather than whichever $\lambda_{\max}$ a source
   publication happened to report.

This notebook adds a holdout that changes only the first. The chromophore database of
Joung *et al.* (*Sci. Data* **2020**, *7*, 295; CC BY 4.0) is an independently assembled
literature compilation whose label is the **first absorption maximum** — a convention much
closer to the heterogeneous reported $\lambda_{\max}$ of the Beard training corpus than a
specified transition assignment is.

The prediction is explicit and falsifiable. If label semantics is the operative confound,
the GP should transfer to this corpus about as well as it transfers to its own novel
scaffolds, despite the corpus being chemically unrelated. If instead the model simply
cannot generalise, it should fail here too.

As in notebook `09`, the published model is **applied, never refitted**, and the
training-fold scaler is used in `transform` mode only.
""",
)

# ---------------------------------------------------------------- 1
add(
    CODE,
    r"""
import json
from collections import defaultdict, deque
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")
UNCHARGER = rdMolStandardize.Uncharger()
RNG = np.random.default_rng(42)

N_BITS = 256
LAMBDA_MIN, LAMBDA_MAX = 200.0, 900.0
MW_MAX, HEAVY_MIN = 1000.0, 3
""",
)

# ---------------------------------------------------------------- 2
add(
    MD,
    r"""
## 12.1 Structure handling

Identical to notebooks `02` and `09`, so that structural identity, the conjugation
descriptor and the fingerprint recipe are the same objects in every phase of the work.

Structural identity is the InChIKey computed after uncharging and removing
stereochemistry. Overlap with training is therefore detected on the same basis used for
the photoswitch holdout, where E/Z isomers of one chromophore appear as separate records.
""",
)

add(
    CODE,
    r'''
def mol_from_smiles(smi):
    """Parse and sanitize a SMILES string; return None if invalid."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def inchikey_nostereo(smi):
    """Structure identity ignoring E/Z and tetrahedral stereo."""
    mol = mol_from_smiles(smi)
    if mol is None:
        return None
    try:
        mol = UNCHARGER.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    Chem.RemoveStereochemistry(mol)
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def largest_conjugated_system(mol):
    """Size of the largest connected conjugated atom system.

    Chromophore colour is governed by the extent of pi delocalisation, so this
    single descriptor carries most of the non-fingerprint signal in the model.
    """
    conj_bonds = [b for b in mol.GetBonds() if b.GetIsConjugated()]
    adj = defaultdict(set)
    for b in conj_bonds:
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[a1].add(a2)
        adj[a2].add(a1)
    visited, largest = set(), 0
    for start in adj:
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        size = 0
        while queue:
            cur = queue.popleft()
            size += 1
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        largest = max(largest, size)
    return largest


def morgan_fp(smi):
    """Binary Morgan fingerprint matching notebook 04c (radius 2, 256 bits)."""
    mol = mol_from_smiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=N_BITS)
    arr = np.zeros((N_BITS,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def murcko(smi):
    """Bemis-Murcko scaffold SMILES, or None if parsing fails."""
    mol = mol_from_smiles(smi)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


def skeleton_key(smi):
    """First block of the InChIKey: the connectivity skeleton alone.

    Collapses tautomers, protonation states and salt forms, so overlap measured
    on this key is an upper bound on structural contamination and a stricter
    check than the stereo-free InChIKey used for removal.
    """
    mol = mol_from_smiles(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol).split("-")[0]
    except Exception:
        return None


print("helpers defined")
''',
)

# ---------------------------------------------------------------- 3
add(
    MD,
    r"""
## 12.2 Curation funnel

The same filters that produced the 6,878-compound modeling set are applied, so the two
corpora are comparable rather than merely adjacent:

- an experimental absorption maximum must be present;
- solution-phase only, since Beard is a solution corpus (Joung flags solid-state records
  by repeating the chromophore in the solvent field);
- the 200–900 nm retention window of notebook `01`;
- the domain filter of notebook `02`, heavy atoms $\geq 3$ and molecular weight
  $\leq 1000$ Da.

Joung reports one record per chromophore–solvent pair. Because the published model has no
solvent input, replicates are mean-collapsed per structure, exactly as duplicate Beard
records were. The spread across solvents for the same chromophore is retained and reported:
it is a solvatochromic noise floor for any solvent-blind model on this corpus.
""",
)

add(
    CODE,
    r"""
raw = pd.read_csv("joung_chromophore_db.csv")
funnel = {"source_records": len(raw), "source_unique_smiles": raw["Chromophore"].nunique()}

df = raw.dropna(subset=["Absorption max (nm)"]).copy()
funnel["with_absorption_max"] = len(df)

solid = df["Solvent"] == df["Chromophore"]
funnel["solid_state_dropped"] = int(solid.sum())
df = df.loc[~solid].copy()

df = df.loc[df["Absorption max (nm)"].between(LAMBDA_MIN, LAMBDA_MAX)].copy()
funnel["in_200_900_window"] = len(df)

uniq = pd.DataFrame({"Chromophore": df["Chromophore"].unique()})
mols = uniq["Chromophore"].map(mol_from_smiles)
funnel["smiles_parse_failures"] = int(mols.isna().sum())
uniq, mols = uniq.loc[mols.notna()].copy(), mols.loc[mols.notna()]

uniq["canonical_smi"] = [Chem.MolToSmiles(m) for m in mols]
uniq["mw"] = [Descriptors.MolWt(m) for m in mols]
uniq["heavy_atoms"] = [m.GetNumHeavyAtoms() for m in mols]
uniq["LargestConjugatedSystemSize"] = [largest_conjugated_system(m) for m in mols]
uniq["inchikey_nostereo"] = uniq["Chromophore"].map(inchikey_nostereo)

df = df.merge(uniq, on="Chromophore", how="inner")
before = len(df)
df = df.loc[(df["mw"] <= MW_MAX) & (df["heavy_atoms"] >= HEAVY_MIN)].copy()
funnel["domain_filter_dropped_records"] = before - len(df)
funnel["after_domain_filter"] = len(df)

grp = df.groupby("inchikey_nostereo")["Absorption max (nm)"]
spread = grp.std().loc[grp.count() > 1].dropna()
funnel["structures_with_multiple_solvents"] = int((grp.count() > 1).sum())
funnel["mean_within_structure_sd_nm"] = float(spread.mean())
funnel["median_within_structure_sd_nm"] = float(spread.median())

agg = (
    df.groupby("inchikey_nostereo")
    .agg(
        canonical_smi=("canonical_smi", "first"),
        lambda_max_exp_nm=("Absorption max (nm)", "mean"),
        n_solvent_records=("Absorption max (nm)", "size"),
        log_epsilon=("log(e/mol-1 dm3 cm-1)", "mean"),
        LargestConjugatedSystemSize=("LargestConjugatedSystemSize", "first"),
        mw=("mw", "first"),
    )
    .reset_index()
)
funnel["unique_structures_collapsed"] = len(agg)

beard = pd.read_csv("beard_model_ready_features.csv")
beard_keys = set(beard["canonical_smi"].map(inchikey_nostereo).dropna())
funnel["beard_modeling_structures"] = len(beard_keys)
in_beard = agg["inchikey_nostereo"].isin(beard_keys)
funnel["overlap_with_training"] = int(in_beard.sum())
funnel["overlap_fraction_of_joung"] = float(in_beard.mean())

holdout = agg.loc[~in_beard].reset_index(drop=True)
holdout["source"] = "Joung et al., Sci. Data 2020 (CC BY 4.0)"
holdout["lambda_max_assignment"] = "first absorption maximum"
funnel["external_holdout_n"] = len(holdout)

print("=== Curation funnel ===")
for k, v in funnel.items():
    print(f"  {k:38s} {v}")
""",
)

# ---------------------------------------------------------------- 4
add(
    MD,
    r"""
### Reading the funnel

Two numbers matter for interpretation.

**Overlap with training is small.** Only a few hundred of the surviving structures were
already in the modeling set, so almost the entire corpus is genuinely unseen. Both
databases are literature-derived, so this is worth checking rather than assuming; the
result is that Beard's auto-extraction over 402,034 documents and Joung's manual curation
sampled largely different chemistry.

**The solvatochromic spread is small relative to the errors of interest.** The mean
within-structure standard deviation across solvents bounds how much of the residual could
be explained by the model's lack of a solvent input, and it is far below the RMSE scale
reported below.
""",
)

# ---------------------------------------------------------------- 5
add(
    MD,
    r"""
## 12.3 Applying the published GP unchanged

Model identity is asserted rather than assumed: the kernel, the feature count and the
training design matrix are printed, and the scaler is used in `transform` mode. Refitting
the scaler would leak holdout feature moments and the model would no longer be the one
evaluated in notebooks `04c`, `08` and `09`.
""",
)

add(
    CODE,
    r"""
gp = joblib.load("gp_full_local_model.joblib")
scaler = joblib.load("scaler_gp.joblib")

print("=== Loaded primary GP ===")
print(f"kernel_                 : {gp.kernel_}")
print(f"n_features_in_ (GP)     : {gp.n_features_in_}")
print(f"n_features_in_ (scaler) : {scaler.n_features_in_}")
print(f"X_train_ shape          : {gp.X_train_.shape}")
print(f"n_restarts_optimizer    : {gp.n_restarts_optimizer}")
print(f"random_state            : {gp.random_state}")

fps = holdout["canonical_smi"].map(morgan_fp)
print(f"\nfingerprint failures    : {int(fps.isna().sum())}")
holdout = holdout.loc[fps.notna()].reset_index(drop=True)
matrix = np.stack(fps.dropna().to_numpy())

X = pd.concat(
    [
        pd.DataFrame(matrix, columns=[f"fp_{i}" for i in range(N_BITS)]),
        holdout[["LargestConjugatedSystemSize"]].reset_index(drop=True),
    ],
    axis=1,
)
print(f"X shape                 : {X.shape}")
print(f"mean bits set           : {matrix.sum(axis=1).mean():.1f} / {N_BITS}")

X_scaled = scaler.transform(X)  # transform only; never refit
y_pred, y_std = gp.predict(X_scaled, return_std=True)
holdout["y_pred"], holdout["y_std"] = y_pred, y_std
holdout["residual"] = y_pred - holdout["lambda_max_exp_nm"]
""",
)

# ---------------------------------------------------------------- 6
add(
    CODE,
    r'''
def bootstrap_metrics(truth, pred, n_boot=2000):
    """Point estimates with bootstrap 95% intervals for RMSE, R2 and bias."""
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    n = len(truth)
    idx = np.arange(n)
    rmse, r2, bias = [], [], []
    for _ in range(n_boot):
        take = RNG.choice(idx, size=n, replace=True)
        t, p = truth[take], pred[take]
        if t.std(ddof=0) == 0:
            continue
        rmse.append(np.sqrt(np.mean((p - t) ** 2)))
        r2.append(r2_score(t, p))
        bias.append((p - t).mean())

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    return {
        "n": n,
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "rmse_ci95": ci(rmse),
        "mae": float(mean_absolute_error(truth, pred)),
        "r2": float(r2_score(truth, pred)),
        "r2_ci95": ci(r2),
        "bias": float((pred - truth).mean()),
        "bias_ci95": ci(bias),
        "pearson_r": float(np.corrcoef(truth, pred)[0, 1]),
        "null_rmse": float(truth.std(ddof=0)),
    }


truth = holdout["lambda_max_exp_nm"].to_numpy(float)
joung_metrics = bootstrap_metrics(truth, y_pred)
joung_metrics["mean_sigma"] = float(y_std.mean())
joung_metrics["sd_sigma"] = float(y_std.std(ddof=0))
joung_metrics["calibration_r"] = float(np.corrcoef(y_std, np.abs(holdout.residual))[0, 1])
joung_metrics["rmse_debiased"] = float(
    np.sqrt(max(joung_metrics["rmse"] ** 2 - joung_metrics["bias"] ** 2, 0.0))
)

print("=== Published GP on the Joung holdout ===")
for k, v in joung_metrics.items():
    print(f"  {k:16s} {v}")
''',
)

# ---------------------------------------------------------------- 7
add(
    MD,
    r"""
## 12.4 Is the corpus actually chemically novel?

A positive result is only interesting if the holdout is genuinely unfamiliar. Bemis-Murcko
scaffolds give the same novelty measure used for the scaffold split in notebook `08`, so
the two can be compared directly.

The relevant benchmark is not "is any of this new" but **"is it newer than the scaffold
split the model already survives"**. If the fraction of unseen scaffolds here exceeds
38.1% — the value for the scaffold-split test set — then this is the harder chemical test
of the two.
""",
)

add(
    CODE,
    r"""
beard_scaffolds = set(s for s in beard["canonical_smi"].map(murcko) if s)
holdout["scaffold"] = holdout["canonical_smi"].map(murcko).fillna("NA")
holdout["scaffold_unseen"] = ~holdout["scaffold"].isin(beard_scaffolds)

print("=== Chemical novelty ===")
print(f"  Beard unique scaffolds          : {len(beard_scaffolds)}")
print(f"  holdout compounds              : {len(holdout)}")
print(f"  holdout unique scaffolds       : {holdout['scaffold'].nunique()}")
print(
    f"  compounds with unseen scaffold : {int(holdout['scaffold_unseen'].sum())}"
    f"  ({100 * holdout['scaffold_unseen'].mean():.1f}%)"
)
print("  scaffold-split test set, for reference: 729 of 1,915 unseen (38.1%)")

# Residual contamination at a granularity stricter than the removal criterion:
# the InChIKey skeleton ignores tautomer, salt and protonation differences that
# the stereo-free key treats as distinct structures.
beard_skeletons = set(s for s in beard["canonical_smi"].map(skeleton_key) if s)
holdout_skeletons = holdout["canonical_smi"].map(skeleton_key)
residual_overlap = int(holdout_skeletons.isin(beard_skeletons).sum())
print("\n=== Residual contamination check (skeleton level) ===")
print(f"  Beard unique skeletons                    : {len(beard_skeletons)}")
print(f"  holdout compounds sharing a skeleton      : {residual_overlap}"
      f"  ({100 * residual_overlap / len(holdout):.2f}%)")
clean = holdout.loc[~holdout_skeletons.isin(beard_skeletons)]
m_clean = bootstrap_metrics(clean["lambda_max_exp_nm"], clean["y_pred"], n_boot=500)
print(f"  GP on skeleton-disjoint holdout (n={m_clean['n']}) : "
      f"RMSE {m_clean['rmse']:.2f}, R2 {m_clean['r2']:+.3f}, bias {m_clean['bias']:+.2f}")

print("\n=== GP by scaffold familiarity ===")
subsets = {}
for name, mask in [
    ("all", np.ones(len(holdout), bool)),
    ("unseen scaffold", holdout["scaffold_unseen"].to_numpy()),
    ("seen scaffold", (~holdout["scaffold_unseen"]).to_numpy()),
]:
    s = holdout.loc[mask]
    m = bootstrap_metrics(s["lambda_max_exp_nm"], s["y_pred"], n_boot=500)
    subsets[name] = m
    print(
        f"  {name:16s} n={m['n']:5d}  RMSE {m['rmse']:6.2f}  R2 {m['r2']:+.3f}"
        f"  bias {m['bias']:+7.2f}  null {m['null_rmse']:6.2f}"
    )
""",
)

# ---------------------------------------------------------------- 8
add(
    MD,
    r"""
## 12.5 The four-tier comparison

Assembling every evaluation regime in order of increasing label mismatch. Tiers 1 and 2
are quoted from notebooks `04c` and `08`; tiers 3 and 4 are computed here and in notebook
`09`.

The axis that varies across tiers 1 to 3 is chemical novelty, while the label convention
is held approximately constant. Only at tier 4 does the label convention change.
""",
)

add(
    CODE,
    r"""
ps = pd.read_csv("external_gp_vs_dft_predictions.csv")
ps_metrics = bootstrap_metrics(ps["lambda_max_exp_nm"], ps["gp_pred"])
ps_metrics["calibration_r"] = float(
    np.corrcoef(ps["gp_std"], np.abs(ps["gp_pred"] - ps["lambda_max_exp_nm"]))[0, 1]
)

unseen = subsets["unseen scaffold"]
tiers = pd.DataFrame(
    [
        {
            "tier": 1, "short": "T1\nBeard\nrandom", "holdout": "Beard, random split",
            "labels": "as training", "n": 1376, "rmse": 92.15, "r2": 0.455,
            "bias": 0.0, "calib_r": 0.059,
            "r2_lo": np.nan, "r2_hi": np.nan, "bias_lo": np.nan, "bias_hi": np.nan,
        },
        {
            "tier": 2, "short": "T2\nBeard\nscaffold", "holdout": "Beard, scaffold split",
            "labels": "as training", "n": 1915, "rmse": 105.93, "r2": 0.241,
            "bias": -10.74, "calib_r": -0.030,
            "r2_lo": np.nan, "r2_hi": np.nan, "bias_lo": np.nan, "bias_hi": np.nan,
        },
        {
            "tier": 3, "short": "T3\nJoung\nunseen scaf.",
            "holdout": "Joung, unseen scaffolds", "labels": "first abs. max",
            "n": unseen["n"], "rmse": unseen["rmse"], "r2": unseen["r2"],
            "bias": unseen["bias"], "calib_r": joung_metrics["calibration_r"],
            "r2_lo": unseen["r2_ci95"][0], "r2_hi": unseen["r2_ci95"][1],
            "bias_lo": unseen["bias_ci95"][0], "bias_hi": unseen["bias_ci95"][1],
        },
        {
            "tier": 4, "short": "T4\nphotoswitch\n$\\pi\\rightarrow\\pi^*$",
            "holdout": "Griffiths photoswitches", "labels": "E-isomer pi-pi*",
            "n": ps_metrics["n"], "rmse": ps_metrics["rmse"], "r2": ps_metrics["r2"],
            "bias": ps_metrics["bias"], "calib_r": ps_metrics["calibration_r"],
            "r2_lo": ps_metrics["r2_ci95"][0], "r2_hi": ps_metrics["r2_ci95"][1],
            "bias_lo": ps_metrics["bias_ci95"][0], "bias_hi": ps_metrics["bias_ci95"][1],
        },
    ]
)
print(tiers.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print()
print("R2 95% CI, tier 3:", [round(v, 3) for v in unseen["r2_ci95"]])
print("R2 95% CI, tier 4:", [round(v, 3) for v in ps_metrics["r2_ci95"]])
print("bias 95% CI, tier 3:", [round(v, 2) for v in unseen["bias_ci95"]])
print("bias 95% CI, tier 4:", [round(v, 2) for v in ps_metrics["bias_ci95"]])
""",
)

# ---------------------------------------------------------------- 9
add(
    MD,
    r"""
## 12.6 Uncertainty calibration is an evaluation-set property

The raw calibration correlation on this corpus is far higher than on the photoswitch
holdout, which invites the conclusion that the model detects chemical extrapolation but
not label mismatch. That conclusion does not survive testing.

Three candidate explanations are separable:

1. **Sample size.** This corpus is roughly eighteen times larger.
2. **Label spread.** Its labels span a much wider wavelength range, and restricted outcome
   variance attenuates correlations for purely statistical reasons.
3. **Genuine risk detection.**

Explanation 1 is tested by subsampling to the photoswitch $n$. Explanation 2 is tested by
resampling so that the holdout reproduces the photoswitch $\lambda_{\max}$ *distribution*,
drawing compounds at random within each wavelength bin so that chemistry stays broad —
a naive contiguous-window match would narrow the label range and the chemistry together,
confounding the very thing being tested.
""",
)

add(
    CODE,
    r'''
def pearson(a, b):
    """Pearson correlation, nan for degenerate input."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fisher_ci(r, n):
    """Fisher z 95% confidence interval for a correlation."""
    if not np.isfinite(r) or n < 4 or abs(r) >= 1:
        return (float("nan"), float("nan"))
    z, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
    return (float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)))


def bin_weights(values, target_density, edges):
    """Sampling weights that reshape a sample toward a target label density."""
    idx = np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)
    observed = np.bincount(idx, minlength=len(edges) - 1).astype(float)
    ratio = np.divide(
        target_density, observed, out=np.zeros_like(target_density, float),
        where=observed > 0,
    )
    w = ratio[idx]
    return w / w.sum() if w.sum() > 0 else np.full(len(values), 1 / len(values))


def matched_calibration(truth, pred, sigma, scaffolds, target_density, edges,
                        n_draw, n_reps=400):
    """Calibration correlation over distribution-matched random subsamples."""
    w = bin_weights(truth, target_density, edges)
    pool = np.flatnonzero(w > 0)
    p = w[pool] / w[pool].sum()
    rs, sds, nscaf = [], [], []
    draw = min(n_draw, len(pool))
    for _ in range(n_reps):
        take = RNG.choice(pool, size=draw, replace=False, p=p)
        rs.append(pearson(sigma[take], np.abs(pred[take] - truth[take])))
        sds.append(truth[take].std(ddof=0))
        nscaf.append(pd.unique(scaffolds[take]).size)
    rs = np.asarray(rs, float)
    rs = rs[np.isfinite(rs)]
    return {
        "n_draw": int(draw), "achieved_label_sd_nm": float(np.mean(sds)),
        "r_mean": float(rs.mean()), "r_sd": float(rs.std(ddof=0)),
        "r_p2.5": float(np.percentile(rs, 2.5)),
        "r_p97.5": float(np.percentile(rs, 97.5)),
        "mean_unique_scaffolds": float(np.mean(nscaf)),
    }


jt = holdout["lambda_max_exp_nm"].to_numpy(float)
jp = holdout["y_pred"].to_numpy(float)
jsig = holdout["y_std"].to_numpy(float)
jscaf = holdout["scaffold"].to_numpy()
pt = ps["lambda_max_exp_nm"].to_numpy(float)

n_ps = len(pt)
raw_r = pearson(jsig, np.abs(jp - jt))
print("=== Raw correlations between predicted sigma and absolute error ===")
print(f"  Joung       n={len(jt):5d}  label sd {jt.std(ddof=0):6.1f}  r = {raw_r:+.3f}"
      f"  CI {tuple(round(v, 3) for v in fisher_ci(raw_r, len(jt)))}")
print(f"  Photoswitch n={n_ps:5d}  label sd {pt.std(ddof=0):6.1f}"
      f"  r = {ps_metrics['calibration_r']:+.3f}"
      f"  CI {tuple(round(v, 3) for v in fisher_ci(ps_metrics['calibration_r'], n_ps))}")

sub_rs = []
for _ in range(400):
    take = RNG.choice(len(jt), size=n_ps, replace=False)
    sub_rs.append(pearson(jsig[take], np.abs(jp[take] - jt[take])))
sub_rs = np.asarray(sub_rs)
print(f"\n=== Explanation 1: sample size (Joung subsampled to n={n_ps}) ===")
print(f"  r = {sub_rs.mean():+.3f} +/- {sub_rs.std(ddof=0):.3f}"
      f"   fraction of draws below 0.10: {(sub_rs < 0.10).mean():.3f}")

edges = np.histogram_bin_edges(jt, bins=24)
ps_hist, _ = np.histogram(pt, bins=edges)
matched = matched_calibration(jt, jp, jsig, jscaf, ps_hist.astype(float), edges, n_ps)
print(f"\n=== Explanation 2: label distribution matched to the photoswitch holdout ===")
print(f"  achieved label sd     : {matched['achieved_label_sd_nm']:.1f} nm"
      f"  (photoswitch {pt.std(ddof=0):.1f})")
print(f"  calibration r         : {matched['r_mean']:+.3f} +/- {matched['r_sd']:.3f}"
      f"   95% range [{matched['r_p2.5']:+.3f}, {matched['r_p97.5']:+.3f}]")
print(f"  scaffolds per draw    : {matched['mean_unique_scaffolds']:.0f} of"
      f" {matched['n_draw']}  (photoswitch holdout has"
      f" {pd.Series(ps['canonical_smi'].map(murcko)).nunique()})")

centres = 0.5 * (edges[:-1] + edges[1:])
mu = jt.mean()
sweep = []
for target_sd in (40, 50, 60, 70, 80, 90, 100, 110):
    dens = np.exp(-0.5 * ((centres - mu) / target_sd) ** 2)
    s = matched_calibration(jt, jp, jsig, jscaf, dens, edges, n_ps, n_reps=200)
    s["requested_label_sd_nm"] = float(target_sd)
    sweep.append(s)

print("\n=== Dose-response: calibration r against label spread ===")
print(f"  {'target sd':>10}{'achieved':>10}{'r':>9}{'r sd':>8}{'scaffolds':>11}")
for s in sweep:
    print(f"  {s['requested_label_sd_nm']:>10.0f}{s['achieved_label_sd_nm']:>10.1f}"
          f"{s['r_mean']:>9.3f}{s['r_sd']:>8.3f}{s['mean_unique_scaffolds']:>11.0f}")
''',
)

# ---------------------------------------------------------------- 10
add(
    MD,
    r"""
### Verdict on uncertainty

Sample size is not the explanation: the correlation is unchanged when this corpus is
subsampled to the photoswitch $n$.

Label distribution largely is. Matching the photoswitch $\lambda_{\max}$ distribution — while
keeping roughly 270 distinct scaffolds per draw, against 34 in the photoswitch holdout —
drops the correlation to about $+0.15$, whose interval overlaps the photoswitch estimate.
The dose-response confirms it: the correlation rises monotonically with label spread.

Two conclusions follow, and the second is the more broadly useful.

The residual $r\approx+0.15$ has a lower bound above zero, so predictive uncertainty is not
strictly uninformative — but a correlation of that size cannot triage predictions, and the
raw $+0.37$ should not be read as applicability-domain detection.

**Calibration correlations are not comparable across evaluation sets with different label
spreads.** Reporting one without the other invites exactly the over-interpretation this
section set out to test.
""",
)

# ---------------------------------------------------------------- 11
add(
    MD,
    r"""
## 12.7 Brightness filtering the holdout

Notebook `11` curated the *training pool* by extinction coefficient and improved transfer
to photoswitches. This corpus reports $\log_{10}\varepsilon$ for a majority of its records,
so the complementary operation — filtering the *holdout* to strongly allowed transitions —
can be tested directly.

These are different operations and the results need not agree; reported here for
completeness rather than as support for the curation argument.
""",
)

add(
    CODE,
    r"""
bright = holdout.dropna(subset=["log_epsilon"])
print(f"epsilon coverage: {holdout['log_epsilon'].notna().mean():.3f}"
      f"  ({len(bright)} of {len(holdout)})")
rows = []
for label, lo in [("all with eps", -np.inf), ("log eps >= 4.0", 4.0), ("log eps >= 4.5", 4.5)]:
    s = bright.loc[bright["log_epsilon"] >= lo]
    if len(s) < 30:
        continue
    r = s["y_pred"].to_numpy() - s["lambda_max_exp_nm"].to_numpy()
    rows.append({
        "subset": label, "n": len(s),
        "rmse": float(np.sqrt((r ** 2).mean())), "bias": float(r.mean()),
        "r2": float(r2_score(s["lambda_max_exp_nm"], s["y_pred"])),
        "exp_mean_nm": float(s["lambda_max_exp_nm"].mean()),
    })
print()
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
""",
)

# ---------------------------------------------------------------- 12
add(
    MD,
    r"""
## 12.8 Figures
""",
)

add(
    CODE,
    r'''
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12})

# --- parity on the second corpus ---------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 6.0))
sc = ax.scatter(jt, jp, c=holdout["LargestConjugatedSystemSize"], s=9, alpha=0.45,
                cmap="viridis", edgecolors="none")
lims = [min(jt.min(), jp.min()) - 20, max(jt.max(), jp.max()) + 20]
ax.plot(lims, lims, "k--", lw=1.4, label="y = x")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Experimental $\\lambda_{max}$ (nm)")
ax.set_ylabel("GP predicted $\\lambda_{max}$ (nm)")
ax.set_title(f"Joung holdout, published GP applied unchanged (n = {len(jt)})")
ax.text(0.03, 0.97,
        f"RMSE = {joung_metrics['rmse']:.1f} nm\n$R^2$ = {joung_metrics['r2']:+.3f}\n"
        f"bias = {joung_metrics['bias']:+.1f} nm",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7"))
ax.legend(loc="lower right")
plt.colorbar(sc, ax=ax, label="Largest conjugated system size")
fig.tight_layout()
fig.savefig("fig_joung_parity.png", dpi=400)
plt.close(fig)

# --- four-tier summary --------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
x = np.arange(len(tiers))
colours = ["#2c7fb8", "#41b6c4", "#7fcdbb", "#d7301f"]


def tier_panel(ax, values, lo, hi, ylabel, title, fmt, pad_frac=0.06):
    """Bar panel with bootstrap intervals where available and non-colliding labels."""
    err = np.vstack([values - np.nan_to_num(lo, nan=values),
                     np.nan_to_num(hi, nan=values) - values])
    ax.bar(x, values, color=colours, edgecolor="black", lw=0.7,
           yerr=err, capsize=4, error_kw={"lw": 1.2, "ecolor": "0.25"})
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers.short, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Headroom first, so label offsets are computed against the final limits.
    top = float(np.nanmax([values.max(), np.nanmax(hi)]))
    bot = float(np.nanmin([values.min(), np.nanmin(lo)]))
    span = top - bot
    ax.set_ylim(bot - pad_frac * span - 0.10 * span, top + pad_frac * span + 0.12 * span)
    off = 0.035 * (ax.get_ylim()[1] - ax.get_ylim()[0])

    for xi, v, l, h in zip(x, values, lo, hi):
        # Place the annotation beyond the error bar, on the far side of the bar
        # from zero, so it never lands on a tick label or a whisker.
        anchor = (h if np.isfinite(h) else v) if v >= 0 else (l if np.isfinite(l) else v)
        y = anchor + off if v >= 0 else anchor - off
        ax.text(xi, y, fmt.format(v), ha="center", fontsize=9,
                va="bottom" if v >= 0 else "top")


tier_panel(
    axes[0], tiers.r2.to_numpy(float), tiers.r2_lo.to_numpy(float),
    tiers.r2_hi.to_numpy(float), "$R^2$ on holdout",
    "(a) Accuracy degrades gracefully, then breaks", "{:+.3f}",
)
tier_panel(
    axes[1], tiers.bias.to_numpy(float), tiers.bias_lo.to_numpy(float),
    tiers.bias_hi.to_numpy(float), "Signed bias (nm)",
    "(b) Only the reassigned label produces a large offset", "{:+.1f}",
)
axes[0].text(0.02, 0.02,
             "error bars: bootstrap 95% CI\n(tiers 1-2 quoted from notebooks 04c, 08)",
             transform=axes[0].transAxes, fontsize=7.5, va="bottom", color="0.35")

fig.tight_layout()
fig.savefig("fig_four_tier_summary.png", dpi=400)
plt.close(fig)

# --- calibration dose-response ------------------------------------------------
fig, ax = plt.subplots(figsize=(6.8, 4.8))
sd = [s["achieved_label_sd_nm"] for s in sweep]
rm = [s["r_mean"] for s in sweep]
rs = [s["r_sd"] for s in sweep]
ax.errorbar(sd, rm, yerr=rs, marker="o", color="#2c7fb8", capsize=3,
            label="Joung, distribution-matched draws")
ax.axhline(raw_r, color="#7fcdbb", ls="-", lw=1.6,
           label=f"Joung, full holdout (r = {raw_r:+.3f})")
ax.axhline(ps_metrics["calibration_r"], color="#d7301f", ls="--", lw=1.6,
           label=f"photoswitch holdout (r = {ps_metrics['calibration_r']:+.3f})")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(pt.std(ddof=0), color="0.5", ls=":", lw=1.2,
           label=f"photoswitch label sd ({pt.std(ddof=0):.0f} nm)")
ax.set_xlabel("Label standard deviation of the evaluation subsample (nm)")
ax.set_ylabel("r(predicted $\\sigma$, |error|)")
ax.set_title("Calibration correlation tracks evaluation-set label spread")
lo_y, hi_y = ax.get_ylim()
ax.set_ylim(lo_y, hi_y + 0.10 * (hi_y - lo_y))  # headroom above the reference line
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
fig.tight_layout()
fig.savefig("fig_calibration_range_dependence.png", dpi=400)
plt.close(fig)

for f in ("fig_joung_parity.png", "fig_four_tier_summary.png",
          "fig_calibration_range_dependence.png"):
    print(f"saved {f}  ({Path(f).stat().st_size / 1024:.0f} kB)")
''',
)

# ---------------------------------------------------------------- 13
add(
    CODE,
    r"""
holdout.drop(columns=["scaffold"]).to_csv("external_validation_joung.csv", index=False)

payload = {
    "curation_funnel": funnel,
    "gp_on_joung_holdout": joung_metrics,
    "by_scaffold_familiarity": subsets,
    "residual_skeleton_overlap": {
        "beard_unique_skeletons": len(beard_skeletons),
        "holdout_sharing_skeleton": residual_overlap,
        "skeleton_disjoint_metrics": m_clean,
    },
    "photoswitch_reference": ps_metrics,
    "four_tier_table": tiers.to_dict(orient="records"),
    "calibration": {
        "joung_raw_r": raw_r,
        "joung_subsampled_r_mean": float(sub_rs.mean()),
        "joung_subsampled_r_sd": float(sub_rs.std(ddof=0)),
        "distribution_matched": matched,
        "label_spread_sweep": sweep,
        "photoswitch_r": ps_metrics["calibration_r"],
    },
    "brightness_filtered_holdout": rows,
}
Path("results_second_external_corpus.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)
print("wrote external_validation_joung.csv and results_second_external_corpus.json")
""",
)

# ---------------------------------------------------------------- 14
add(
    MD,
    r"""
## 12.9 Conclusions

**Chemical novelty alone does not break the model.** On a corpus assembled independently
of Beard, with a larger fraction of unseen Bemis-Murcko scaffolds than the scaffold-split
test set the model already survives, the GP retains positive $R^2$ and a small signed
offset. Overlap with training is negligible, and residual contamination at the
tautomer-insensitive skeleton level is a single compound.

**Reassigning the label does break it.** The photoswitch holdout is chemically narrower and
much smaller, yet it alone produces negative $R^2$ and a large systematic red shift. The
confidence intervals on $R^2$ for the two external corpora do not overlap, and the bias
magnitudes differ roughly sixfold.

Together these separate the two candidate explanations that notebook `11` could only
address indirectly. The operative confound is what the benchmark labels denote, not how
novel the chemistry is.

**The uncertainty result is a caution, not a capability.** The higher calibration
correlation on this corpus is largely attributable to its wider and heavier-tailed label
distribution. Matched on that distribution it falls to about $+0.15$, overlapping the
photoswitch estimate. Predictive uncertainty remains too weak to triage predictions in any
external regime tested, and calibration correlations should not be compared across
evaluation sets of differing label spread.

**Limitation.** Joung labels are manually curated to a stated convention, the first
absorption maximum, whereas Beard labels are auto-extracted. Joung's labels are therefore
cleaner as well as differently defined, and this analysis cannot separate label *quality*
from label *convention*. What it does establish is that a chemically unfamiliar corpus with
a compatible label convention is predicted acceptably, while a chemically narrow corpus
with an incompatible one is not.
""",
)


def build() -> dict[str, Any]:
    """Assemble the notebook JSON structure."""
    cells: list[dict[str, Any]] = []
    for kind, body in CELLS:
        cell: dict[str, Any] = {
            "cell_type": kind,
            "metadata": {},
            "source": body.splitlines(keepends=True),
        }
        if kind == CODE:
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.20"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "12_second_external_corpus.ipynb"
    out.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"wrote {out.name} with {len(CELLS)} cells")
