"""Assemble composite manuscript Figures 5–8 from notebooks 09–13 artifacts.

| Manuscript | Notebook | Output PNG |
|------------|----------|------------|
| Figure 5   | 09       | Figure5_external_validation.png |
| Figure 6   | 12       | Figure6_second_external_corpus.png |
| Figure 7   | 13       | Figure7_external_tddft_benchmark.png |
| Figure 8   | 11       | Figure8_label_curation.png |

Figure 5 — external photoswitch holdout and coverage-matched TD-DFT comparison.
Figure 6 — second external corpus: four-tier generalisation audit and uncertainty
           range-dependence controls.
Figure 7 — published ωB97X-D3 vertical excitations versus the GP on the mixed-
           convention holdout, with the full comparison matrix.
Figure 8 — label-curation transfer: pool displacement, curated vs size-matched
           control, and in-class recalibration floor.

Run from the repository root (this directory). Requires the CSV/JSON outputs of
notebooks 09–13.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update(
    {
        "savefig.dpi": 400,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "axes.axisbelow": True,
    }
)

HERE = Path(__file__).parent
PANEL_KW = dict(fontsize=16, fontweight="bold", va="top", ha="left")

CAM = "CAM-B3LYP/6-31G**"
PBE = "PBE0"
DFT_COLS = {
    PBE: "PBE0 DFT E isomer pi-pi* wavelength in nm",
    CAM: "CAM-B3LYP/6-31G** DFT E isomer pi-pi* wavelength in nm",
}


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def split_error(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Return (systematic offset, residual scatter) summing in quadrature to RMSE."""
    bias = float(np.mean(np.asarray(p, float) - np.asarray(y, float)))
    total = rmse(y, p)
    return abs(bias), float(np.sqrt(max(total**2 - bias**2, 0.0)))


# --------------------------------------------------------------------- data
pred = pd.read_csv(HERE / "external_gp_vs_dft_predictions.csv")
ext = pd.read_csv(HERE / "external_validation.csv")
pred = pred.merge(
    ext[["inchikey_nostereo", "LargestConjugatedSystemSize"]],
    on="inchikey_nostereo",
    how="left",
    validate="one_to_one",
)
curation = json.loads((HERE / "results_label_curation_transfer.json").read_text())
second = json.loads((HERE / "results_second_external_corpus.json").read_text())
tddft = json.loads((HERE / "results_external_tddft_benchmark.json").read_text())

# In-corpus quantum-chemistry comparisons, quoted from the leakage-free holdouts of
# notebook 04c and verified in _repro_final_report.json. They are quoted rather than
# recomputed because they require the Beard computed columns, not redistributed here.
INTERNAL_QC = {
    "tddft": {"n": 36, "gp": 87.28, "qc": 108.70, "method": "TD-DFT"},
    "stda": {"n": 1011, "gp": 81.60, "qc": 117.10, "method": "sTDA"},
}

y_all = pred["lambda_max_exp_nm"].to_numpy()
gp_all = pred["gp_pred"].to_numpy()
conj_all = pred["LargestConjugatedSystemSize"].to_numpy()


# ===================================================================== FIG 5
def figure5() -> None:
    fig = plt.figure(figsize=(13.0, 10.4))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.24)

    # ---- (a) GP on the full external holdout
    ax = fig.add_subplot(gs[0, 0])
    bias_all = float(np.mean(gp_all - y_all))
    lo, hi = 240, 660
    sc = ax.scatter(y_all, gp_all, c=conj_all, cmap="viridis", s=34, alpha=0.85,
                    edgecolor="k", linewidth=0.35)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.9, label="$y=x$")
    ax.plot([lo, hi], [lo + bias_all, hi + bias_all], "k:", lw=1.9,
            label=f"$y=x{bias_all:+.0f}$ nm")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"experimental $\lambda_{\max}$ (nm)")
    ax.set_ylabel(r"GP predicted $\lambda_{\max}$ (nm)")
    ax.set_title(f"GP, full external holdout (n = {len(y_all)})\n"
                 f"RMSE = {rmse(y_all, gp_all):.1f} nm, "
                 f"$R^2$ = {r2(y_all, gp_all):.2f}, bias = {bias_all:+.0f} nm")
    ax.legend(loc="upper left")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("conjugated-system size", fontsize=11)
    ax.text(-0.16, 1.06, "a", transform=ax.transAxes, **PANEL_KW)

    # ---- (b,c) coverage-matched GP vs CAM-B3LYP
    sub = pred.dropna(subset=[DFT_COLS[CAM]])
    y = sub["lambda_max_exp_nm"].to_numpy()
    gp_s = sub["gp_pred"].to_numpy()
    cam_s = sub[DFT_COLS[CAM]].to_numpy()
    bias_s = float(np.mean(gp_s - y))

    for col, pvals, title, shift in (
        (1, gp_s, "GP (this work)", True),
        (2, cam_s, CAM, False),
    ):
        ax = fig.add_subplot(gs[0, 1] if col == 1 else gs[1, 0])
        ax.scatter(y, pvals, c=sub["LargestConjugatedSystemSize"], cmap="viridis",
                   s=34, alpha=0.85, edgecolor="k", linewidth=0.35)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.9, label="$y=x$")
        if shift:
            ax.plot([lo, hi], [lo + bias_s, hi + bias_s], "k:", lw=1.9,
                    label=f"$y=x{bias_s:+.0f}$ nm")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(r"experimental $\lambda_{\max}$ (nm)")
        ax.set_ylabel(r"predicted $\lambda_{\max}$ (nm)")
        ax.set_title(f"{title}, coverage-matched (n = {len(y)})\n"
                     f"RMSE = {rmse(y, pvals):.1f} nm, $R^2$ = {r2(y, pvals):.2f}")
        ax.legend(loc="upper left")
        ax.text(-0.16, 1.06, "b" if col == 1 else "c", transform=ax.transAxes, **PANEL_KW)

    # ---- (d) total error vs error remaining after the offset is removed for free.
    # RMSE and scatter combine in quadrature, so these are shown as grouped bars
    # rather than stacked segments.
    ax = fig.add_subplot(gs[1, 1])
    pos = 0.0
    xs: list[float] = []
    labels: list[str] = []
    ns: list[int] = []
    for name in (PBE, CAM):
        s = pred.dropna(subset=[DFT_COLS[name]])
        ys = s["lambda_max_exp_nm"].to_numpy()
        ns.append(len(s))
        for label, vals, colour in (
            (name.split("/")[0], s[DFT_COLS[name]].to_numpy(), "#2c7fb8"),
            ("GP", s["gp_pred"].to_numpy(), "#d95f0e"),
        ):
            total = rmse(ys, vals)
            _, scatter = split_error(ys, vals)
            ax.bar(pos - 0.19, total, width=0.36, color=colour)
            ax.bar(pos + 0.19, scatter, width=0.36, color=colour, alpha=0.45, hatch="//")
            ax.text(pos - 0.19, total + 1.4, f"{total:.1f}", ha="center", fontsize=10.5)
            ax.text(pos + 0.19, scatter + 1.4, f"{scatter:.1f}", ha="center", fontsize=10.5)
            xs.append(pos)
            labels.append(label)
            pos += 1.05
        pos += 0.6
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("external RMSE (nm)")
    ax.set_ylim(0, 104)
    ax.set_title("Total error vs error after removing each method's offset\n"
                 f"(left: {PBE}, n = {ns[0]};  right: CAM-B3LYP, n = {ns[1]})")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#5a5a5a"),
        plt.Rectangle((0, 0), 1, 1, color="#5a5a5a", alpha=0.45, hatch="//"),
    ]
    ax.legend(handles, ["RMSE", "residual scatter (offset removed)"],
              loc="upper left", fontsize=9.5)
    ax.text(-0.14, 1.06, "d", transform=ax.transAxes, **PANEL_KW)

    out = HERE / "Figure5_external_validation.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out.name}")
    plt.close(fig)


# ===================================================================== FIG 6
def figure6() -> None:
    fig = plt.figure(figsize=(16.4, 5.6))
    gs = fig.add_gridspec(1, 3, wspace=0.28)

    # ---- (a) pool displacement
    ax = fig.add_subplot(gs[0, 0])
    dists = {r["pool"]: r for r in curation["label_distributions"]}
    pool_df = pd.read_csv(HERE / "beard_model_ready_features.csv")
    cleaned = pd.read_csv(HERE / "beard_uvvis_cleaned.csv")
    merged = pool_df.merge(
        cleaned[["canonical_smi", "extinction"]].drop_duplicates("canonical_smi"),
        on="canonical_smi", how="left", validate="one_to_one",
    )
    eps = pd.to_numeric(merged["extinction"], errors="coerce").fillna(-1.0)
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    azo_pat = Chem.MolFromSmarts("[#6]N=N[#6]")
    is_azo = merged["canonical_smi"].map(
        lambda s: bool((m := Chem.MolFromSmiles(s)) and m.HasSubstructMatch(azo_pat))
    )

    bins = np.linspace(200, 760, 57)
    target = ext["lambda_max_exp_nm"]
    ax.hist(target, bins=bins, density=True, color="#2ca02c", alpha=0.34,
            label=f"external target (n={len(target)})")
    series = {
        "Beard, all": (merged["lambda_max_exp_nm"], "#7f7f7f"),
        "bright ($\\varepsilon\\geq$10k)": (merged.loc[eps >= 10000, "lambda_max_exp_nm"], "#2c7fb8"),
        "azo only": (merged.loc[is_azo, "lambda_max_exp_nm"], "#d62728"),
    }
    for label, (s, colour) in series.items():
        ax.hist(s, bins=bins, density=True, histtype="step", lw=2.3, color=colour,
                label=f"{label} (n={len(s)})")
        ax.axvline(s.mean(), color=colour, ls=":", lw=1.5)
    ax.axvline(target.mean(), color="#2ca02c", ls="--", lw=2.1)
    ax.set_xlabel(r"experimental $\lambda_{\max}$ (nm)")
    ax.set_ylabel("density")
    ax.set_xlim(200, 760)
    ax.set_title("Training-pool displacement\n(dotted lines: pool means)")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.text(-0.17, 1.07, "a", transform=ax.transAxes, **PANEL_KW)

    # ---- (b) curated vs size-matched control
    ax = fig.add_subplot(gs[0, 1])
    exps = curation["experiments"]
    names = list(exps)
    width = 0.2
    base = np.arange(len(names))
    for k, (metric, off, hatch) in enumerate(
        [("rmse", -1.5, None), ("bias", 0.5, "//")]
    ):
        cur_v = [exps[n]["curated"]["external_318"][metric] for n in names]
        ctl_v = [exps[n][f"control_{'rmse' if metric == 'rmse' else 'bias'}"][0] for n in names]
        ctl_e = [exps[n][f"control_{'rmse' if metric == 'rmse' else 'bias'}"][1] for n in names]
        ax.bar(base + (off) * width, cur_v, width, color="#2c7fb8", hatch=hatch,
               label=f"curated, {metric.upper() if metric == 'rmse' else 'bias'}")
        ax.bar(base + (off + 1) * width, ctl_v, width, yerr=ctl_e, capsize=5,
               color="#9e9e9e", hatch=hatch,
               label=f"control, {metric.upper() if metric == 'rmse' else 'bias'}")
    ax.set_xticks(base)
    ax.set_xticklabels([f"bright\n($\\varepsilon\\geq$10k, n={exps[names[0]]['train_n']})",
                        f"azo only\n(n={exps[names[1]]['train_n']})"])
    ax.set_ylabel("nm")
    ax.set_ylim(0, 108)
    ax.set_title("Curated vs size-matched random control\n(external holdout, n = 318)")
    ax.legend(loc="upper left", fontsize=8.8, ncol=2)
    ax.text(-0.17, 1.07, "b", transform=ax.transAxes, **PANEL_KW)

    # ---- (c) recalibration floor
    ax = fig.add_subplot(gs[0, 2])
    cam_rmse = curation["dft_reference"][CAM]["rmse"]
    colours = {"published GP (notebook 10)": "#d95f0e",
               "curated GP: bright (eps >= 10k)": "#2c7fb8"}
    pretty = {"published GP (notebook 10)": "published GP",
              "curated GP: bright (eps >= 10k)": "curated GP ($\\varepsilon\\geq$10k)"}
    for label, curve in curation["calibration_curves"].items():
        pts = sorted(curve.values(), key=lambda v: v["k"])
        ks = [v["k"] for v in pts]
        mu = np.array([v["rmse_mean"] for v in pts])
        sd = np.array([v["rmse_std"] for v in pts])
        ax.errorbar(ks, mu, yerr=sd, marker="o", ms=6, lw=2.1, capsize=4,
                    color=colours[label], label=pretty[label])
        ax.fill_between(ks, mu - sd, mu + sd, color=colours[label], alpha=0.15)
    ax.axhline(cam_rmse, color="#2ca02c", ls="--", lw=2.3,
               label=f"CAM-B3LYP = {cam_rmse:.1f} nm")
    ax.axhline(curation["dft_reference"][PBE]["rmse"], color="#2ca02c", ls=":", lw=2.0,
               label=f"PBE0 = {curation['dft_reference'][PBE]['rmse']:.1f} nm")
    ax.set_xlabel("in-class calibration measurements $k$")
    ax.set_ylabel("RMSE on held-out molecules (nm)")
    ax.set_ylim(0, 86)
    ax.set_title("In-class recalibration floor\n(CAM-B3LYP subset, n = 114)")
    ax.legend(loc="lower left", fontsize=9.5)
    ax.text(-0.17, 1.07, "c", transform=ax.transAxes, **PANEL_KW)

    out = HERE / "Figure8_label_curation.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out.name}")
    plt.close(fig)


# ===================================================================== FIG 7
def figure7() -> None:
    """Four-tier generalisation audit plus the calibration range-dependence control.

    Panels (b) and (c) carry the central claim: prediction quality declines
    gracefully as chemical novelty increases while the label convention holds,
    and breaks only when the benchmark reassigns the label to one specified
    electronic transition. Panel (d) is the control establishing that the
    apparently better uncertainty calibration on the second corpus is a property
    of its label distribution rather than evidence of domain awareness.
    """
    joung = pd.read_csv(HERE / "external_validation_joung.csv")
    tiers = second["four_tier_table"]
    sweep = second["calibration"]["label_spread_sweep"]
    matched = second["calibration"]["distribution_matched"]
    jm = second["gp_on_joung_holdout"]
    psr = second["photoswitch_reference"]

    fig = plt.figure(figsize=(14.0, 10.6))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.38)

    # ---- (a) parity on the second corpus. Hexbin, because 5,811 overlapping
    # points hide their own density in a scatter plot.
    ax = fig.add_subplot(gs[0, 0])
    y = joung["lambda_max_exp_nm"].to_numpy()
    p = joung["y_pred"].to_numpy()
    lo, hi = 190, 900
    hb = ax.hexbin(y, p, gridsize=54, cmap="viridis", mincnt=1, bins="log",
                   extent=(lo, hi, lo, hi))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.9, label="$y=x$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"experimental $\lambda_{\max}$ (nm)")
    ax.set_ylabel(r"GP predicted $\lambda_{\max}$ (nm)")
    ax.set_title(f"GP on second external corpus (n = {jm['n']})\n"
                 f"RMSE = {jm['rmse']:.1f} nm, $R^2$ = {jm['r2']:+.2f}, "
                 f"bias = {jm['bias']:+.1f} nm")
    ax.legend(loc="upper left")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("compounds per bin", fontsize=11)
    ax.text(-0.16, 1.06, "a", transform=ax.transAxes, **PANEL_KW)

    # ---- (b,c) the four-tier audit
    colours = ["#2c7fb8", "#41b6c4", "#7fcdbb", "#d7301f"]
    xs = np.arange(len(tiers))
    xlabels = [t["short"] for t in tiers]

    for panel, key, ylabel, title, fmt in (
        ("b", "r2", "$R^2$ on holdout",
         "Accuracy degrades gracefully, then breaks", "{:+.3f}"),
        ("c", "bias", "signed bias (nm)",
         "Only the reassigned label produces a large offset", "{:+.1f}"),
    ):
        ax = fig.add_subplot(gs[0, 1] if panel == "b" else gs[1, 0])
        vals = np.array([t[key] for t in tiers], float)
        los = np.array([t[f"{key}_lo"] if t[f"{key}_lo"] is not None else np.nan
                        for t in tiers], float)
        his = np.array([t[f"{key}_hi"] if t[f"{key}_hi"] is not None else np.nan
                        for t in tiers], float)
        err = np.vstack([vals - np.nan_to_num(los, nan=vals),
                         np.nan_to_num(his, nan=vals) - vals])
        ax.bar(xs, vals, color=colours, edgecolor="black", lw=0.7, yerr=err,
               capsize=4, error_kw={"lw": 1.2, "ecolor": "0.25"})
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=9.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        top = float(np.nanmax([vals.max(), np.nanmax(his)]))
        bot = float(np.nanmin([vals.min(), np.nanmin(los)]))
        span = top - bot
        ax.set_ylim(bot - 0.16 * span, top + 0.18 * span)
        off = 0.035 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        for xi, v, l, h in zip(xs, vals, los, his):
            anchor = (h if np.isfinite(h) else v) if v >= 0 else (l if np.isfinite(l) else v)
            ax.text(xi, anchor + off if v >= 0 else anchor - off, fmt.format(v),
                    ha="center", fontsize=10,
                    va="bottom" if v >= 0 else "top")
        if panel == "b":
            ax.text(0.02, 0.03, "error bars: bootstrap 95% CI\n"
                                "tiers 1-2 quoted from notebooks 04c, 08",
                    transform=ax.transAxes, fontsize=8, va="bottom", color="0.35")
        ax.text(-0.16, 1.06, panel, transform=ax.transAxes, **PANEL_KW)

    # ---- (d) calibration is a property of the evaluation set
    ax = fig.add_subplot(gs[1, 1])
    sd = [s["achieved_label_sd_nm"] for s in sweep]
    rm = [s["r_mean"] for s in sweep]
    rs = [s["r_sd"] for s in sweep]
    ax.errorbar(sd, rm, yerr=rs, marker="o", ms=6, lw=2.1, capsize=4,
                color="#2c7fb8", label="label-distribution-matched draws")
    ax.axhline(second["calibration"]["joung_raw_r"], color="#7fcdbb", lw=2.3,
               label=f"second corpus, full "
                     f"(r = {second['calibration']['joung_raw_r']:+.3f})")
    ax.axhline(psr["calibration_r"], color="#d7301f", ls="--", lw=2.3,
               label=f"photoswitch holdout (r = {psr['calibration_r']:+.3f})")
    ax.axhline(0, color="black", lw=0.9)
    ax.axvline(psr["null_rmse"], color="0.45", ls=":", lw=1.8,
               label=f"photoswitch label SD ({psr['null_rmse']:.0f} nm)")
    ax.annotate(
        f"matched to the photoswitch\ndistribution: r = {matched['r_mean']:+.3f}",
        xy=(matched["achieved_label_sd_nm"], matched["r_mean"]),
        xytext=(matched["achieved_label_sd_nm"] - 26, matched["r_mean"] + 0.115),
        fontsize=9, arrowprops=dict(arrowstyle="->", lw=1.2, color="0.3"),
    )
    ax.set_xlabel("label SD of the evaluation subsample (nm)")
    ax.set_ylabel(r"$r$(predicted $\sigma$, $|$error$|$)")
    ax.set_title("Uncertainty calibration tracks evaluation-set\nlabel spread, "
                 "not chemical familiarity")
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y, hi_y + 0.12 * (hi_y - lo_y))
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.text(-0.16, 1.06, "d", transform=ax.transAxes, **PANEL_KW)

    out = HERE / "Figure6_second_external_corpus.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out.name}")
    plt.close(fig)


# ===================================================================== FIG 8
def figure8() -> None:
    """The external quantum-chemical benchmark that closes the comparison matrix.

    Panels (a) and (b) show the same molecules and the same experimental labels
    scored by published wB97X-D3 vertical excitations and by the GP. The visual
    difference is the argument: an offset but tight diagonal band against a
    near-horizontal cloud. Panel (c) separates each method's removable offset
    from its irreducible scatter, since only the former can be corrected by a
    user holding a few reference measurements. Panel (d) assembles every
    GP-versus-quantum-chemistry comparison in the paper and shows that the
    model's advantage is confined to in-corpus holdouts.
    """
    gvt = pd.read_csv(HERE / "external_gp_vs_tddft_joung.csv")
    board = tddft["scoreboard"]
    s1_key = "TD-DFT: lowest excitation (S1)"
    gp_key = "GP (this work)"
    n_matched = tddft["matching"]["matched"]
    null = tddft["null_rmse"]

    fig = plt.figure(figsize=(14.0, 10.8))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.30)

    # ---- (a,b) parity, on shared axes so the two are directly comparable
    lo, hi = 250, 800
    truth = gvt["lambda_max_exp_nm"].to_numpy(float)
    panels = (
        ("a", gs[0, 0], gvt["tddft_nm__lowest_excitation_(S1)"].to_numpy(float),
         r"$\omega$B97X-D3 vertical $S_1$", "Blues", board[s1_key]),
        ("b", gs[0, 1], gvt["y_pred"].to_numpy(float),
         "published GP", "Oranges", board[gp_key]),
    )
    for tag, cell, values, name, cmap, stats in panels:
        ax = fig.add_subplot(cell)
        hb = ax.hexbin(truth, values, gridsize=46, cmap=cmap, mincnt=1, bins="log",
                       extent=(lo, hi, lo, hi))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.8, label="$y=x$")
        ax.plot([lo, hi], [lo + stats["bias"], hi + stats["bias"]], color="crimson",
                lw=1.8, label=f"offset {stats['bias']:+.0f} nm")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(r"experimental $\lambda_{\max}$ (nm)")
        ax.set_ylabel(rf"{name} $\lambda_{{\max}}$ (nm)")
        ax.set_title(f"{name}\nRMSE = {stats['rmse']:.0f} nm, scatter = "
                     f"{stats['scatter']:.0f} nm, $r$ = {stats['pearson_r']:.2f}")
        ax.legend(loc="upper left", framealpha=0.94)
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("compounds per bin", fontsize=11)
        ax.text(-0.17, 1.06, tag, transform=ax.transAxes, **PANEL_KW)

    # ---- (c) where each method's error lives. Offset and scatter combine in
    # quadrature, so they are drawn side by side rather than stacked.
    ax = fig.add_subplot(gs[1, 0])
    keys = [s1_key, "TD-DFT: lowest bright (f>=0.01)", "TD-DFT: brightest overall", gp_key]
    labels = ["TD-DFT\n$S_1$", "TD-DFT\nlowest bright", "TD-DFT\nbrightest", "published\nGP"]
    xs = np.arange(len(keys))
    series = (
        ("RMSE, raw", [board[k]["rmse"] for k in keys], "#34495e"),
        ("|offset|, removable", [abs(board[k]["bias"]) for k in keys], "#a9cce3"),
        ("scatter, irreducible", [board[k]["scatter"] for k in keys], "#c0392b"),
        ("RMSE after affine fit", [board[k]["cv_affine_rmse"] for k in keys], "#7dcea0"),
    )
    width = 0.2
    for j, (label, values, colour) in enumerate(series):
        pos = xs + (j - 1.5) * width
        ax.bar(pos, values, width, label=label, color=colour, edgecolor="black", lw=0.5)
        for xi, value in zip(pos, values):
            ax.text(xi, value + 2.0, f"{value:.0f}", ha="center", fontsize=8.5)
    ax.axhline(null, ls=":", color="black", lw=1.4, label=f"null model ({null:.0f} nm)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("error (nm)")
    ax.set_title("The GP has almost no correctable component")
    ax.legend(fontsize=9, ncol=2, loc="upper left", framealpha=0.95)
    ax.set_ylim(0, max(max(v) for _, v, _ in series) * 1.44)
    ax.text(-0.17, 1.06, "c", transform=ax.transAxes, **PANEL_KW)

    # ---- (d) the complete comparison matrix
    ax = fig.add_subplot(gs[1, 1])
    ps = json.loads((HERE / "results_external_dft_benchmark.json").read_text())
    ps_cam = ps["comparisons"][CAM]
    ps_gp = ps_cam["gp_matched"]["rmse"]
    ps_qc = ps_cam["dft"]["rmse"]
    ps_n = ps_cam["dft"]["n"]

    rows = [
        (f"Beard\nTD-DFT\nin-corpus, mixed\nn = {INTERNAL_QC['tddft']['n']}",
         INTERNAL_QC["tddft"]["gp"], INTERNAL_QC["tddft"]["qc"], False),
        (f"Beard\nsTDA\nin-corpus, mixed\nn = {INTERNAL_QC['stda']['n']:,}",
         INTERNAL_QC["stda"]["gp"], INTERNAL_QC["stda"]["qc"], False),
        (f"Joung\n$\\omega$B97X-D3\nexternal, mixed\nn = {n_matched:,}",
         board[gp_key]["rmse"], board[s1_key]["rmse"], True),
        (f"Photoswitch\nCAM-B3LYP\nexternal, specified\nn = {ps_n}",
         ps_gp, ps_qc, True),
    ]
    xs = np.arange(len(rows))
    width = 0.34
    gp_vals = [r[1] for r in rows]
    qc_vals = [r[2] for r in rows]
    ax.bar(xs - width / 2, gp_vals, width, label="published GP",
           color="#d68910", edgecolor="black", lw=0.6)
    ax.bar(xs + width / 2, qc_vals, width, label="quantum chemistry",
           color="#2471a3", edgecolor="black", lw=0.6)

    for xi, (_, gp_v, qc_v, external) in zip(xs, rows):
        for dx, value in ((-width / 2, gp_v), (width / 2, qc_v)):
            ax.text(xi + dx, value + 2.5, f"{value:.0f}", ha="center", fontsize=9.5)
        winner = "GP" if gp_v < qc_v else "QC"
        ax.text(xi, max(gp_v, qc_v) + 15, f"{winner} wins",
                ha="center", fontsize=9.5, fontweight="bold",
                color="#b9770e" if winner == "GP" else "#1a5276")
        if external:
            ax.axvspan(xi - 0.5, xi + 0.5, color="0.88", zorder=0)

    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8.2)
    ax.set_ylabel("RMSE (nm)")
    ax.set_title("The GP's advantage does not leave its own corpus")
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    ax.set_ylim(0, max(qc_vals + gp_vals) * 1.38)
    ax.text(0.02, 0.965, "shaded: external holdouts", transform=ax.transAxes,
            fontsize=8.6, ha="left", va="top", color="0.35")
    ax.text(-0.17, 1.06, "d", transform=ax.transAxes, **PANEL_KW)

    out = HERE / "Figure7_external_tddft_benchmark.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out.name}")
    plt.close(fig)


if __name__ == "__main__":
    figure5()
    figure6()
    figure7()
    figure8()
    for f in ("Figure5_external_validation.png", "Figure6_second_external_corpus.png",
              "Figure7_external_tddft_benchmark.png",
              "Figure8_label_curation.png"):
        p = HERE / f
        print(f"{f}: {p.stat().st_size / 1024:.0f} kB")
