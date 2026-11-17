"""A szakdolgozat ábrái és összefoglaló táblázatai.

Nem futtatja újra a modelleket: a results/embeddings/ beágyazásait és a
results/tables/ eredményeit olvassa be.

Futtatás:  python -m src.figures
Kimenet:   results/figures/*.png  (300 dpi)
           results/tables/osszehasonlitas.xlsx
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy.spatial.distance import pdist

from src import config, data_prep

# Megjelenés

COLORS = {"Eredeti": "#898781", "PCA": "#2a78d6", "MDS": "#eb6834",
          "t-SNE": "#1baf7a", "Autoencoder": "#eda100"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

MARKERS = {"concrete": "o", "fat": "s", "ess": "^"}
DS_LABEL = {"concrete": "Concrete", "fat": "Fat", "ess": "ESS"}
ALL = ["Eredeti"] + config.METHODS

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "legend.frameon": False,
})


def hu(x: float, d: int = 2) -> str:
    """Szám tizedesvesszővel (magyar formátum)."""
    return f"{x:.{d}f}".replace(".", ",")


def hu_axis(axis, d: int = 1, fixed: bool = False) -> None:
    """Tizedesvesszős tengelyfeliratok; fixed=True: mindig d tizedesjegy (a 0 kivételével)."""
    axis.set_major_formatter(FuncFormatter(
        lambda v, _: "0" if v == 0 else hu(v, d) if (fixed or v % 1) else f"{v:.0f}"))


def hu_log_axis(axis, ticks=None) -> None:
    """Logaritmikus tengely 1, 2, 5, 10, 20 ... jelölésekkel, tizedesvesszővel."""
    if ticks is None:
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    else:
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis.set_minor_formatter(NullFormatter())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))


def save(fig, name: str) -> None:
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURE_DIR / f"{name}.png", facecolor="white")
    plt.close(fig)
    print(f"  {name}.png")


# Adatok

def load_results():
    comp = pd.read_csv(config.TABLE_DIR / "comparison.csv")
    dist = {n: pd.read_csv(config.TABLE_DIR / f"distance_corr_{n}.csv", index_col=0)
            for n in config.DATASETS}
    comp["rho_Eredeti"] = [dist[d].loc["Eredeti", m] for d, m in zip(comp.dataset, comp.method)]
    return comp, dist


def load_matrices(name: str):
    """Standardizált X, a beágyazások és a célváltozó egy adatbázisra."""
    X, y, _ = data_prep.load_dataset(name)
    q = data_prep.kaiser_q(X)
    emb = {"Eredeti": data_prep.standardize(X)}
    for m in config.METHODS:
        emb[m] = data_prep.load_embedding(name, m, q)
    return X, y, emb


def vif_vector(A: np.ndarray) -> np.ndarray:
    return np.diag(np.linalg.inv(np.corrcoef(A, rowvar=False)))


def panel_title(name: str, comp: pd.DataFrame) -> str:
    row = comp[(comp.dataset == name) & (comp.method == "Eredeti")].iloc[0]
    q = comp[(comp.dataset == name) & (comp.method == "PCA")].q.iloc[0]
    return f"{DS_LABEL[name]}  (p = {row.q}, q = {q})"


def vif_thresholds(ax, horizontal: bool = True) -> None:
    for v, label in [(5, "VIF = 5"), (10, "VIF = 10")]:
        line = ax.axhline if horizontal else ax.axvline
        line(v, color="#d03b3b", lw=0.9, alpha=0.7, zorder=1)


# Ábrák

def fig_vif_original(matrices) -> None:
    """1. ábra: az eredeti magyarázóváltozók VIF-értékei – a kiinduló probléma."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 7.5), gridspec_kw={"wspace": 0.9})
    for ax, name in zip(axes, config.DATASETS):
        X, _, emb = matrices[name]
        vif = pd.Series(vif_vector(emb["Eredeti"]), index=X.columns).sort_values()
        colors = [COLORS["Eredeti"] if v < 5 else "#ec835a" if v < 10 else "#d03b3b"
                  for v in vif]
        ax.barh(vif.index, vif.values, color=colors, height=0.7)
        ax.set_xscale("log")
        hu_log_axis(ax.xaxis, ticks=[1, 2, 5, 10, 20, 50, 100])
        ax.set_xlim(0.8, max(15, vif.max() * 1.6))
        vif_thresholds(ax, horizontal=False)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8 if len(vif) > 15 else 9)
        ax.set_title(f"{DS_LABEL[name]}  (max VIF = {hu(vif.max(), 1)})", loc="left")
        ax.set_xlabel("VIF (logaritmikus skála)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (COLORS["Eredeti"], "#ec835a", "#d03b3b")]
    fig.legend(handles, ["VIF < 5", "5 ≤ VIF < 10", "VIF ≥ 10"], ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Az eredeti magyarázóváltozók varianciainflációs tényezői",
                 x=0.06, ha="left", fontweight="bold", fontsize=13)
    save(fig, "01_vif_eredeti_valtozok")


def fig_vif_methods(matrices, comp) -> None:
    """2. ábra: VIF minden változóra / új tengelyre, módszerenként."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    rng = np.random.default_rng(config.RANDOM_STATE)
    for ax, name in zip(axes, config.DATASETS):
        _, _, emb = matrices[name]
        for i, m in enumerate(ALL):
            v = vif_vector(emb[m])
            jitter = rng.uniform(-0.18, 0.18, len(v)) if len(v) > 2 else np.zeros(len(v))
            ax.scatter(i + jitter, v, s=26, color=COLORS[m], alpha=0.75,
                       edgecolor="white", linewidth=0.8, zorder=3)
            ax.plot([i - 0.3, i + 0.3], [v.max()] * 2, color=INK, lw=1.6, zorder=4)
            ax.text(i, v.max() * 1.25, hu(v.max(), 2), ha="center", fontsize=8, color=INK2)
        ax.set_yscale("log")
        hu_log_axis(ax.yaxis)
        ax.set_ylim(0.7, 400)
        vif_thresholds(ax)
        ax.set_xticks(range(len(ALL)), ALL, rotation=30, ha="right")
        ax.grid(axis="x", visible=False)
        ax.set_title(panel_title(name, comp), loc="left")
    axes[0].set_ylabel("VIF (logaritmikus skála)")
    axes[-1].text(4.45, 10, "VIF = 10", color="#d03b3b", fontsize=8, va="bottom", ha="right")
    axes[-1].text(4.45, 5, "VIF = 5", color="#d03b3b", fontsize=8, va="bottom", ha="right")
    fig.suptitle("Multikollinearitás a dimenziócsökkentés előtt és után "
                 "(pont: egy változó/tengely VIF-je, vonal: maximum)",
                 x=0.06, ha="left", fontweight="bold", fontsize=12, y=1.02)
    save(fig, "02_vif_modszerenkent")


def fig_condition_number(comp) -> None:
    """3. ábra: a korrelációs mátrix kondíciószáma."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, name in zip(axes, config.DATASETS):
        d = comp[comp.dataset == name].set_index("method").loc[ALL]
        ax.bar(range(len(ALL)), d.condition_number, color=[COLORS[m] for m in ALL], width=0.62)
        for i, v in enumerate(d.condition_number):
            ax.text(i, v * 1.15, hu(v, 1), ha="center", fontsize=8, color=INK2)
        ax.set_yscale("log")
        hu_log_axis(ax.yaxis)
        ax.set_ylim(0.8, 8000)
        ax.axhline(30, color="#d03b3b", lw=0.9, alpha=0.7)
        ax.set_xticks(range(len(ALL)), ALL, rotation=30, ha="right")
        ax.grid(axis="x", visible=False)
        ax.set_title(panel_title(name, comp), loc="left")
    axes[0].set_ylabel("Kondíciószám (λmax / λmin, log)")
    axes[-1].text(4.45, 30, "κ = 30", color="#d03b3b", fontsize=8, va="bottom", ha="right")
    fig.suptitle("A korrelációs mátrix kondíciószáma", x=0.06, ha="left",
                 fontweight="bold", fontsize=12, y=1.02)
    save(fig, "03_kondicioszam")


def fig_explanatory(comp) -> None:
    """4. ábra: korrigált R² adatbázisonként, a megőrzött arány a címkén."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, name in zip(axes, config.DATASETS):
        d = comp[comp.dataset == name].set_index("method").loc[ALL]
        ax.bar(range(len(ALL)), d.R2_adj, color=[COLORS[m] for m in ALL], width=0.62)
        top = d.R2_adj.max()
        for i, (m, r) in enumerate(d.iterrows()):
            label = hu(r.R2_adj, 3) if m == "Eredeti" else f"{hu(r.R2_adj, 3)}\n({r.R2_retained:.0%})"
            ax.text(i, r.R2_adj + top * 0.02, label, ha="center", va="bottom",
                    fontsize=8, color=INK2)
        ax.set_ylim(0, top * 1.25)
        hu_axis(ax.yaxis, 2, fixed=True)
        ax.set_xticks(range(len(ALL)), ALL, rotation=30, ha="right")
        ax.grid(axis="x", visible=False)
        kind = "McFadden-féle korr. pszeudo-R²" if config.DATASETS[name]["task"] == "classification" \
            else "Korrigált R²"
        ax.set_ylabel(kind)
        ax.set_title(panel_title(name, comp), loc="left")
    fig.suptitle("Magyarázóerő: korrigált R² (zárójelben az eredeti modell R²-ének megőrzött hányada)",
                 x=0.06, ha="left", fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "04_magyarazoero")


def fig_retained(comp) -> None:
    """5. ábra: megőrzött magyarázóerő, a három adatbázis egy közös skálán."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    names = list(config.DATASETS)
    width = 0.19
    for j, m in enumerate(config.METHODS):
        vals = [comp[(comp.dataset == n) & (comp.method == m)].R2_retained.iloc[0] for n in names]
        x = np.arange(len(names)) + (j - 1.5) * (width + 0.02)
        ax.bar(x, vals, width=width, color=COLORS[m], label=m)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", fontsize=7.5, color=INK2)
    ax.axhline(1, color=COLORS["Eredeti"], lw=1)
    ax.text(len(names) - 0.5, 1.01, "Eredeti = 100%", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(names)), [DS_LABEL[n] for n in names])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 1.12)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("R² / R²(Eredeti)")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.1))
    ax.set_title("Az eredeti modell magyarázóerejének megőrzött hányada", loc="left")
    save(fig, "05_megorzott_magyarazoero")


def fig_distance_heatmaps(dist, comp) -> None:
    """6. ábra: a távolságmátrixok Spearman-korrelációja minden módszerpárra."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"wspace": 0.45})
    for ax, name in zip(axes, config.DATASETS):
        c = dist[name].loc[ALL, ALL].to_numpy()
        mask = np.triu(np.ones_like(c, bool), k=1)
        im = ax.imshow(np.ma.masked_array(c, mask), cmap=BLUES, vmin=0.5, vmax=1)
        for i in range(len(ALL)):
            for j in range(i + 1):
                ax.text(j, i, hu(c[i, j], 2), ha="center", va="center", fontsize=9,
                        color="white" if c[i, j] > 0.78 else INK)
        ax.set_xticks(range(len(ALL)), ALL, rotation=35, ha="right")
        ax.set_yticks(range(len(ALL)), ALL)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(panel_title(name, comp), loc="left")
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cb.set_label("Spearman-féle ρ")
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: hu(v, 1)))
    cb.outline.set_visible(False)
    fig.suptitle("A páronkénti távolságok rangkorrelációja (távolságmátrixok egyezése)",
                 x=0.1, ha="left", fontweight="bold", fontsize=12, y=1.02)
    save(fig, "06_tavolsagmatrix_korrelacio")


def fig_distance_original(comp) -> None:
    """7. ábra: mennyire őrzi meg az egyes módszer az eredeti távolságszerkezetet."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    names = list(config.DATASETS)
    width = 0.19
    for j, m in enumerate(config.METHODS):
        vals = [comp[(comp.dataset == n) & (comp.method == m)].rho_Eredeti.iloc[0] for n in names]
        x = np.arange(len(names)) + (j - 1.5) * (width + 0.02)
        ax.bar(x, vals, width=width, color=COLORS[m], label=m)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.012, hu(v, 2), ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(names)), [DS_LABEL[n] for n in names])
    ax.set_ylim(0, 1.08)
    hu_axis(ax.yaxis, 1, fixed=True)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Spearman-féle ρ az eredeti távolságokkal")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.1))
    ax.set_title("Az eredeti (standardizált) tér távolságszerkezetének megőrzése", loc="left")
    save(fig, "07_tavolsag_megorzes")


def fig_shepard(matrices, comp, n_pairs: int = 4000) -> None:
    """8. ábra: Shepard-diagramok – eredeti vs. beágyazott páronkénti távolságok."""
    names = list(config.DATASETS)
    fig, axes = plt.subplots(len(names), len(config.METHODS), figsize=(13, 9.5))
    rng = np.random.default_rng(config.RANDOM_STATE)
    for r, name in enumerate(names):
        _, _, emb = matrices[name]
        d0 = pdist(emb["Eredeti"])
        idx = rng.choice(len(d0), size=min(n_pairs, len(d0)), replace=False)
        for c, m in enumerate(config.METHODS):
            ax = axes[r, c]
            d1 = pdist(emb[m])
            ax.scatter(d0[idx], d1[idx], s=3, color=COLORS[m], alpha=0.25, linewidth=0)
            rho = comp[(comp.dataset == name) & (comp.method == m)].rho_Eredeti.iloc[0]
            ax.text(0.04, 0.94, f"ρ = {hu(rho, 3)}", transform=ax.transAxes, va="top",
                    fontsize=9, fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
            hu_axis(ax.xaxis, 1)
            hu_axis(ax.yaxis, 1)
            if r == 0:
                ax.set_title(m, color=INK)
            if c == 0:
                ax.set_ylabel(f"{DS_LABEL[name]}\nbeágyazott távolság", color=INK)
            if r == len(names) - 1:
                ax.set_xlabel("eredeti távolság")
    fig.suptitle(f"Shepard-diagramok (adatbázisonként {n_pairs} véletlen megfigyeléspár)",
                 x=0.06, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, "08_shepard_diagramok")


def fig_embedding_fat(matrices) -> None:
    """9. ábra: a Fat adatbázis kétdimenziós beágyazásai a testzsír szerint színezve."""
    _, y, emb = matrices["fat"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw={"wspace": 0.38})
    for ax, m in zip(axes, config.METHODS):
        H = emb[m]
        sc = ax.scatter(H[:, 0], H[:, 1], c=y, cmap=BLUES, s=16, edgecolor="white",
                        linewidth=0.4, vmin=y.min(), vmax=y.max())
        ax.set_title(m, loc="left", color=COLORS[m] if m != "Autoencoder" else "#b87d00")
        ax.set_xlabel(f"{m} 1. tengely")
        ax.set_ylabel(f"{m} 2. tengely")
        hu_axis(ax.xaxis, 1)
        hu_axis(ax.yaxis, 1)
    cb = fig.colorbar(sc, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("Testzsír (%)")
    cb.outline.set_visible(False)
    fig.suptitle("A Fat adatbázis kétdimenziós beágyazásai (q = 2), testzsír szerint színezve",
                 x=0.06, ha="left", fontweight="bold", fontsize=12, y=1.03)
    save(fig, "09_beagyazasok_fat")


def fig_tradeoff(comp) -> None:
    """10. ábra: multikollinearitás-csökkentés vs. megőrzött magyarázóerő."""
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    fig.subplots_adjust(right=0.76)
    z = {"Eredeti": 3, "t-SNE": 4, "MDS": 5, "Autoencoder": 6, "PCA": 7}  # PCA legfelül
    for _, r in comp.iterrows():
        ax.scatter(r.max_VIF, r.R2_retained, s=90, marker=MARKERS[r.dataset],
                   color=COLORS[r.method], edgecolor="white", linewidth=1.5, zorder=z[r.method])
    ax.annotate("Fat: PCA ≈ Autoencoder", xy=(1.03, 0.752), xytext=(1.5, 0.83), fontsize=8,
                color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ax.set_xscale("log")
    hu_log_axis(ax.xaxis)
    ax.set_xlim(0.8, 250)
    ax.set_ylim(0.2, 1.08)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.axvline(10, color="#d03b3b", lw=0.9, alpha=0.7)
    ax.text(10.5, 0.22, "VIF = 10", color="#d03b3b", fontsize=8)
    ax.annotate("ideális tartomány:\nalacsony VIF, magas R²", xy=(0.85, 1.04), fontsize=8.5,
                color=INK2, va="top")
    ax.set_xlabel("Legnagyobb VIF (logaritmikus skála)")
    ax.set_ylabel("Megőrzött magyarázóerő (R² / R²(Eredeti))")
    m_handles = [Line2D([], [], marker="o", ls="", color=COLORS[m], markersize=8, label=m) for m in ALL]
    d_handles = [Line2D([], [], marker=MARKERS[d], ls="", color=MUTED, markersize=8,
                        label=DS_LABEL[d]) for d in config.DATASETS]
    fig.legend(handles=m_handles, title="Módszer", loc="center left", bbox_to_anchor=(0.77, 0.66))
    fig.legend(handles=d_handles, title="Adatbázis", loc="center left", bbox_to_anchor=(0.77, 0.27))
    ax.set_title("Kompromisszum: multikollinearitás megszüntetése vs. magyarázóerő", loc="left")
    save(fig, "10_kompromisszum")


# Rangsor és nyertes

CRITERIA = {  # oszlop: (megjelenített név, nagyobb a jobb?)
    "R2_adj": ("Magyarázóerő\n(korr. R²)", True),
    "max_VIF": ("Multikollinearitás\n(max VIF)", False),
    "rho_Eredeti": ("Távolságmegőrzés\n(ρ az eredetivel)", True),
}


def ranking(comp: pd.DataFrame):
    """Módszerek rangsorolása adatbázisonként és kritériumonként (1 = legjobb).

    Kiegészítő pontszám: min-max normált érték (1 = a legjobb, 0 = a leggyengébb
    módszer az adott adatbázison), a VIF-et logaritmikus skálán normálva.
    """
    d = comp[comp.method != "Eredeti"].copy()
    d["log_VIF"] = np.log10(d.max_VIF)
    ranks, scores = [], []
    for name, g in d.groupby("dataset", sort=False):
        g = g.set_index("method")
        for col, (_, higher) in CRITERIA.items():
            ranks.append(g[col].rank(ascending=not higher, method="min").rename((name, col)))
            v = g["log_VIF"] if col == "max_VIF" else g[col]
            s = (v - v.min()) / (v.max() - v.min())
            scores.append((s if higher else 1 - s).rename((name, col)))
    ranks = pd.concat(ranks, axis=1).loc[config.METHODS]
    scores = pd.concat(scores, axis=1).loc[config.METHODS]
    ranks.columns = pd.MultiIndex.from_tuples(ranks.columns, names=["dataset", "criterion"])
    scores.columns = pd.MultiIndex.from_tuples(scores.columns, names=["dataset", "criterion"])

    summary = pd.DataFrame({
        **{f"átl. rang – {CRITERIA[c][0].split(chr(10))[0]}": ranks.xs(c, level=1, axis=1).mean(axis=1)
           for c in CRITERIA},
        "Átlagos rang (összesen)": ranks.mean(axis=1),
        "Normált pontszám (0–1)": scores.mean(axis=1),
    })
    summary["Végső helyezés"] = summary["Átlagos rang (összesen)"].rank(method="min").astype(int)
    return ranks, scores, summary.sort_values("Átlagos rang (összesen)")


def fig_ranking(comp) -> None:
    """11. ábra: rangsor-hőtérkép és összesített átlagos rang – a nyertes."""
    ranks, _, summary = ranking(comp)
    cols = [(n, c) for c in CRITERIA for n in config.DATASETS]
    R = ranks[cols].to_numpy()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 4.3),
                                 gridspec_kw={"width_ratios": [3.2, 1.3], "wspace": 0.35})
    cmap = LinearSegmentedColormap.from_list("rank", ["#184f95", "#5598e7", "#b7d3f6", "#f0efec"])
    ax.imshow(R, cmap=cmap, vmin=1, vmax=4, aspect="auto")
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            ax.text(j, i, f"{R[i, j]:.0f}.", ha="center", va="center", fontsize=10,
                    color="white" if R[i, j] <= 2 else INK, fontweight="bold")
    ax.set_xticks(range(len(cols)), [DS_LABEL[n] for n, _ in cols], fontsize=9)
    ax.set_yticks(range(len(config.METHODS)), config.METHODS)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for k, c in enumerate(CRITERIA):
        mid = k * len(config.DATASETS) + (len(config.DATASETS) - 1) / 2
        ax.text(mid, -0.75, CRITERIA[c][0], ha="center", va="bottom", fontsize=9.5,
                fontweight="bold")
        if k:
            ax.axvline(k * len(config.DATASETS) - 0.5, color="white", lw=4)
    ax.set_title("Helyezés kritériumonként és adatbázisonként (1. = legjobb)",
                 loc="left", pad=42)

    order = summary.index[::-1]
    vals = summary.loc[order, "Átlagos rang (összesen)"]
    bx.barh(range(len(order)), vals, color=[COLORS[m] for m in order], height=0.6)
    for i, (m, v) in enumerate(vals.items()):
        bx.text(v + 0.05, i, hu(v, 2), va="center", fontsize=9,
                fontweight="bold" if m == summary.index[0] else "normal")
    bx.set_yticks(range(len(order)), order)
    bx.set_xlim(0, 4.3)
    bx.set_xlabel("Átlagos rang (kisebb = jobb)")
    bx.grid(axis="y", visible=False)
    bx.set_title(f"Összesítve – nyertes: {summary.index[0]}", loc="left", pad=42)
    save(fig, "11_rangsor_nyertes")
    return summary


# Táblázatok (Excel)

def write_tables(comp, dist) -> None:
    ranks, scores, summary = ranking(comp)
    task_note = {n: " (McFadden)" if config.DATASETS[n]["task"] == "classification" else ""
                 for n in config.DATASETS}
    base = comp.assign(Adatbázis=comp.dataset.map(DS_LABEL), Módszer=comp.method)

    t1 = base.assign(**{"R²": comp.R2, "Korr. R²": comp.R2_adj,
                        "Megőrzött R² (%)": comp.R2_retained * 100,
                        "Szignifikáns együtthatók (%)": comp.sig_share * 100,
                        "Megjegyzés": comp.dataset.map(lambda n: "pszeudo-R²" + task_note[n]
                                                       if task_note[n] else "")})
    t1 = t1[["Adatbázis", "Módszer", "q", "R²", "Korr. R²", "Megőrzött R² (%)",
             "Szignifikáns együtthatók (%)", "Megjegyzés"]]

    t2 = base.assign(**{"Max VIF": comp.max_VIF, "Átlagos VIF": comp.mean_VIF,
                        "Kondíciószám": comp.condition_number,
                        "Értelmezhetőség (R²)": comp.interpretability_R2})
    t2 = t2[["Adatbázis", "Módszer", "q", "Max VIF", "Átlagos VIF", "Kondíciószám",
             "Értelmezhetőség (R²)"]]

    t3 = comp[comp.method != "Eredeti"].pivot(index="method", columns="dataset",
                                              values="rho_Eredeti")
    t3 = t3.loc[config.METHODS, list(config.DATASETS)].rename(columns=DS_LABEL)
    t3["Átlag"] = t3.mean(axis=1)
    t3.index.name = "Módszer"

    rank_long = ranks.copy()
    rank_long.columns = [f"{DS_LABEL[n]} – {CRITERIA[c][0].split(chr(10))[0]}" for n, c in ranks.columns]
    rank_long.index.name = "Módszer"
    summary.index.name = "Módszer"

    path = config.TABLE_DIR / "osszehasonlitas.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        t1.to_excel(xw, sheet_name="1_Magyarazoero", index=False)
        t2.to_excel(xw, sheet_name="2_Multikollinearitas", index=False)
        t3.to_excel(xw, sheet_name="3_Tavolsagmegorzes")
        for n in config.DATASETS:
            dist[n].loc[ALL, ALL].to_excel(xw, sheet_name=f"4_Tavolsagmatrix_{DS_LABEL[n]}")
        rank_long.to_excel(xw, sheet_name="5_Rangok")
        summary.to_excel(xw, sheet_name="6_Osszesites")

        from openpyxl.styles import Font
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 34)
                for c in col[1:]:
                    if isinstance(c.value, float):
                        c.number_format = "0.0" if ("%" in str(col[0].value)
                                                    or "Kondíció" in str(col[0].value)) else "0.000"
            ws.freeze_panes = "B2"
        ws = xw.book["6_Osszesites"]
        for c in ws[2]:
            c.font = Font(bold=True)
    print(f"  {path.relative_to(config.ROOT)}")


def main() -> None:
    comp, dist = load_results()
    matrices = {n: load_matrices(n) for n in config.DATASETS}

    print(f"Ábrák -> {config.FIGURE_DIR.relative_to(config.ROOT)}")
    fig_vif_original(matrices)
    fig_vif_methods(matrices, comp)
    fig_condition_number(comp)
    fig_explanatory(comp)
    fig_retained(comp)
    fig_distance_heatmaps(dist, comp)
    fig_distance_original(comp)
    fig_shepard(matrices, comp)
    fig_embedding_fat(matrices)
    fig_tradeoff(comp)
    summary = fig_ranking(comp)

    print("Táblázatok:")
    write_tables(comp, dist)

    print("\nÖsszesített rangsor:")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
