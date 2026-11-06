"""Ábrák a szakdolgozathoz (results/figures, 300 dpi PNG).

Futtatás (a run_experiments után):  python -m src.plots
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src import config
from src.diagnostics import load_dataset

# Rögzített színhozzárendelés: a módszer színe minden ábrán ugyanaz.
METHOD_COLORS = {"PCA": "#2a78d6", "MDS": "#eb6834", "t-SNE": "#1baf7a", "Autoencoder": "#eda100"}
REF_COLOR = "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])
SEQ_HEAT = LinearSegmentedColormap.from_list("heat_blue", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"])
DS_LABEL = {"concrete": "Concrete (beton)", "fat": "FAT (testzsír)", "ess": "ESS2020"}
TARGET_LABEL = {"concrete": "Nyomószilárdság (MPa)", "fat": "Testzsír (%)", "ess": "Beoltatná magát"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 110, "savefig.dpi": 300,
    "savefig.bbox": "tight", "legend.frameon": False,
})


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / name, facecolor="white")
    plt.close(fig)
    print("  ->", name)


def scree_plots():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, ds in zip(axes, config.DATASETS):
        ev = pd.read_csv(config.TABLE_DIR / f"eigenvalues_{ds}.csv")
        q = int((ev.eigenvalue > 1).sum())
        colors = ["#2a78d6" if i < q else "#b7d3f6" for i in range(len(ev))]
        ax.bar(ev.component, ev.eigenvalue, color=colors, width=0.75, edgecolor="white", linewidth=1)
        ax.axhline(1, color=INK2, lw=1, ls="--")
        ax.text(len(ev) + 0.4, 1.05, "Kaiser: λ = 1", ha="right", va="bottom", color=INK2, fontsize=8)
        cum = ev.cumulative.iloc[q - 1]
        ax.set_title(f"{DS_LABEL[ds]}\nq = {q} komponens, kumulált variancia {cum:.0%}", loc="left")
        ax.set_xlabel("Főkomponens")
        ax.set_ylabel("Sajátérték")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "scree_plots.png")


def embedding_scatter(setting_q=2):
    for ds in config.DATASETS:
        X, y, cfg = load_dataset(ds)
        if not all((config.RESULTS_DIR / "embeddings" / f"{ds}_{m}_q{setting_q}.csv").exists()
                   for m in config.METHODS):
            print(f"  (kihagyva: {ds} q={setting_q} beágyazások hiányoznak)")
            continue
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
        for ax, m in zip(axes, config.METHODS):
            path = config.RESULTS_DIR / "embeddings" / f"{ds}_{m}_q{setting_q}.csv"
            if not path.exists():
                ax.set_visible(False)
                continue
            H = pd.read_csv(path).values
            if cfg["task"] == "classification":
                for val, col, lab in [(0, "#eb6834", "Nem"), (1, "#2a78d6", "Igen")]:
                    s = y.values == val
                    ax.scatter(H[s, 0], H[s, 1], s=9, c=col, alpha=0.6, linewidths=0, label=lab)
            else:
                sc = ax.scatter(H[:, 0], H[:, 1], s=10, c=y.values, cmap=SEQ, linewidths=0)
            ax.set_title(m, loc="left", color=METHOD_COLORS[m], fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(GRID)
        if cfg["task"] == "classification":
            axes[-1].legend(title=TARGET_LABEL[ds], loc="center left", bbox_to_anchor=(1.02, 0.5),
                            markerscale=2)
        else:
            cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01)
            cb.set_label(TARGET_LABEL[ds]); cb.outline.set_visible(False)
        fig.suptitle(f"{DS_LABEL[ds]}: kétdimenziós beágyazások (q = 2), színezés a célváltozó szerint",
                     x=0.01, y=1.06, ha="left", color=INK, fontsize=11)
        _save(fig, f"embedding_2d_{ds}.png")


def correlation_heatmaps(res):
    for ds in config.DATASETS:
        X, _, _ = load_dataset(ds)
        main = res[(res.dataset == ds) & (res.setting == "main")]
        if main.empty:
            continue
        q = int(main.q.iloc[0])
        mats = [("Eredeti változók", np.abs(np.corrcoef(X.values, rowvar=False)))]
        for m in config.METHODS:
            H = pd.read_csv(config.RESULTS_DIR / "embeddings" / f"{ds}_{m}_q{q}.csv").values
            mats.append((m, np.abs(np.corrcoef(H, rowvar=False))))
        fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), gridspec_kw={"width_ratios": [1.6, 1, 1, 1, 1]})
        for ax, (title, M) in zip(axes, mats):
            im = ax.imshow(M, cmap=SEQ_HEAT, vmin=0, vmax=1)
            ax.set_title(title, loc="left", color=METHOD_COLORS.get(title, INK),
                         fontweight="bold" if title in METHOD_COLORS else "normal")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
        cb.set_label("|r|"); cb.outline.set_visible(False)
        fig.suptitle(f"{DS_LABEL[ds]}: abszolút korrelációs mátrixok (q = {q})", x=0.01, y=1.08, ha="left",
                     color=INK, fontsize=11)
        _save(fig, f"corr_heatmaps_{ds}.png")


def _dot_panel(ax, sub, col, ref, title, xlabel, log=False, err=None):
    methods = [m for m in config.METHODS if m in sub.method.values]
    ypos = np.arange(len(methods))[::-1]
    for yp, m in zip(ypos, methods):
        v = sub.loc[sub.method == m, col].iloc[0]
        e = sub.loc[sub.method == m, err].iloc[0] if err and err in sub else None
        if e is not None and not np.isnan(e):
            ax.errorbar(v, yp, xerr=e, color=METHOD_COLORS[m], lw=2, capsize=0)
        ax.scatter(v, yp, s=70, color=METHOD_COLORS[m], edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(f"{v:.3g}", (v, yp), xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=8, color=INK2)
    if log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if ref is not None and not np.isnan(ref):
        ax.axvline(ref, color=REF_COLOR, lw=1.5, ls="--")
        lo, hi = ax.get_xlim()
        frac = ((np.log10(ref) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) if log
                else (ref - lo) / (hi - lo))
        left_side = frac > 0.5
        ax.annotate(f" eredeti: {ref:.3g} " , (ref, ypos.max() + 0.55), ha="right" if left_side else "left",
                    fontsize=8, color=INK2, annotation_clip=False)
    ax.set_yticks(ypos, methods)
    ax.set_ylim(-0.6, len(methods) - 0.2)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def comparison_panels(res, setting="main"):
    specs = [
        ("max_VIF", "Max. VIF (log skála)", True, None, "A) Multikollinearitás"),
        ("trustworthiness", "Trustworthiness (k = 10)", False, None, "B) Lokális szerkezet"),
        ("linear_recon_R2", "Visszaállítható variancia", False, None, "B) Globális információ"),
        (None, None, False, None, "C) Előrejelzés (CV)"),
        ("sign_flip_rate", "Bootstrap előjelváltási arány", False, None, "D) Becslési stabilitás"),
    ]
    dss = list(config.DATASETS)
    fig, axes = plt.subplots(len(dss), len(specs), figsize=(17, 2.6 * len(dss)))
    for i, ds in enumerate(dss):
        sub = res[(res.dataset == ds) & (res.setting == setting)]
        ref = res[(res.dataset == ds) & (res.method == "Eredeti")]
        if sub.empty:
            continue
        task = config.DATASETS[ds]["task"]
        for j, (col, xl, log, err, head) in enumerate(specs):
            ax = axes[i, j]
            if col is None:
                col = "cv_R2" if task == "regression" else "cv_AUC"
                xl = "CV R²" if task == "regression" else "CV AUC"
                err = col + "_sd"
            r = ref[col].iloc[0] if col in ref and not ref.empty else None
            _dot_panel(ax, sub, col, r, "", xl, log, err)
            axes[0, j].set_title(head, loc="left", pad=14)
            if j == 0:
                ax.set_ylabel(f"{DS_LABEL[ds]}\nq = {int(sub.q.iloc[0])}", color=INK, fontsize=10)
            else:
                ax.set_yticklabels([])
    fig.tight_layout()
    _save(fig, f"comparison_{setting}.png")


def rank_heatmap(ranks, setting="main"):
    r = ranks[ranks.setting == setting].pivot(index="method", columns="dataset", values="rank_overall")
    r = r.reindex(index=config.METHODS, columns=[d for d in config.DATASETS if d in r.columns])
    r["Átlag"] = r.mean(axis=1)
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    cmap = LinearSegmentedColormap.from_list("rank", ["#1c5cab", "#86b6ef", "#f0efec"])
    ax.imshow(r.values, cmap=cmap, vmin=1, vmax=4, aspect="auto")
    for (a, b), v in np.ndenumerate(r.values):
        ax.text(b, a, f"{v:.2f}", ha="center", va="center", color="white" if v < 2 else INK, fontsize=9)
    ax.set_xticks(range(r.shape[1]), [DS_LABEL.get(c, c) for c in r.columns], rotation=0, fontsize=8)
    ax.set_yticks(range(r.shape[0]), r.index)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Átlagos rang a mutatócsoportokon (1 = legjobb)", loc="left")
    _save(fig, f"rank_heatmap_{setting}.png")


def main():
    res = pd.read_csv(config.TABLE_DIR / "results_all.csv")
    ranks = pd.read_csv(config.TABLE_DIR / "ranks.csv")
    scree_plots()
    embedding_scatter(2)
    correlation_heatmaps(res)
    for s in ["main", "q2"]:
        if (res.setting == s).any():
            comparison_panels(res, s)
            rank_heatmap(ranks, s)


if __name__ == "__main__":
    main()
