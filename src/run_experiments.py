"""A 3 adatbázis x 4 módszer = 12 modell futtatása és értékelése.

Dimenziószám (q):
  - fő beállítás: Kaiser-kritérium a korrelációs mátrixon (sajátérték > 1, 2.2 fejezet),
    ugyanaz a q mind a négy módszerre -> igazságos összehasonlítás
  - kiegészítő beállítás: q = 2 (vizualizációs eset, ahol az MDS és a t-SNE "otthon van")

Referencia: OLS / logisztikus regresszió az összes eredeti (multikollineáris) változón.

A beágyazások a results/embeddings mappába kerülnek; újrafuttatáskor ezeket újrahasznosítja
(gyorsítótár). Teljes újraszámolás:  python -m src.run_experiments --recompute
Csak egy adatbázis:                   python -m src.run_experiments --datasets fat
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.diagnostics import load_dataset
from src.evaluation import (METRIC_DIRECTION, collinearity_metrics, cv_inductive, cv_transductive,
                            inference_metrics, structure_metrics)
from src.reducers import AutoencoderReducer, make_reducer

EMB_DIR = config.RESULTS_DIR / "embeddings"


def kaiser_q(Z: np.ndarray) -> int:
    ev = np.linalg.eigvalsh(np.corrcoef(Z, rowvar=False))
    return max(2, int((ev > 1).sum()))


def get_embedding(ds, method, q, Z, recompute):
    path = EMB_DIR / f"{ds}_{method}_q{q}.csv"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists() and not recompute:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return pd.read_csv(path).values, meta
    red = make_reducer(method, q)
    H = red.fit_transform(Z)
    meta = {"fit_time_s": red.fit_time_, **red.info_}
    pd.DataFrame(H, columns=[f"{method}_{i + 1}" for i in range(q)]).to_csv(path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return H, meta


def evaluate(ds, method, q, setting, X, Z, y, task, H, meta, inductive=True):
    row = {"dataset": ds, "method": method, "q": q, "setting": setting}
    t0 = time.perf_counter()
    row.update(collinearity_metrics(H))
    if method != "Eredeti":
        row.update(structure_metrics(Z, H))
    row.update(cv_transductive(H, y, task))
    row.update(inference_metrics(H, y, task))
    if inductive and method in ("PCA", "Autoencoder"):
        if method == "PCA":
            factory = lambda: make_reducer("PCA", q)
        else:  # CV-n belül kevesebb újraindítás a futási idő miatt
            factory = lambda: AutoencoderReducer(q, n_restarts=2)
        row.update(cv_inductive(X, y, task, factory))
    row["fit_time_s"] = meta.get("fit_time_s", np.nan)
    for k in ["cumulative_explained", "kl_divergence", "raw_stress", "val_r2_best", "recon_mse", "architecture"]:
        if k in meta:
            row[k] = meta[k]
    print(f"  {ds:9s} {method:12s} q={q:<3d} kész ({time.perf_counter() - t0:.1f}s értékelés, "
          f"{row['fit_time_s']:.1f}s illesztés)")
    return row


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Módszerenkénti átlagos rang adatbázisonként és beállításonként (1 = legjobb).
    Az 'Eredeti' referenciasor nem vesz részt a rangsorban."""
    out = []
    groups = {
        "rank_collinearity": ["max_VIF", "condition_number", "max_abs_corr"],
        "rank_structure": ["trustworthiness", "shepard_spearman", "stress1", "linear_recon_R2"],
        "rank_prediction": ["cv_R2", "cv_RMSE", "cv_AUC", "cv_logloss"],
        "rank_stability": ["share_sig_coefs", "sign_flip_rate", "boot_coef_sd"],
    }
    for (ds, setting), g in df[df.method != "Eredeti"].groupby(["dataset", "setting"]):
        g = g.copy()
        for grp, cols in groups.items():
            ranks = []
            for c in cols:
                if c in g and g[c].notna().all():
                    ranks.append(g[c].rank(ascending=METRIC_DIRECTION[c] < 0, method="average"))
            g[grp] = pd.concat(ranks, axis=1).mean(axis=1) if ranks else np.nan
        g["rank_overall"] = g[list(groups)].mean(axis=1)
        out.append(g[["dataset", "setting", "method"] + list(groups) + ["rank_overall"]])
    return pd.concat(out).sort_values(["setting", "dataset", "rank_overall"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(config.DATASETS))
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--no-q2", action="store_true", help="a q=2 kiegészítő futás kihagyása")
    args = ap.parse_args()

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in args.datasets:
        Xdf, ydf, cfg = load_dataset(ds)
        X, y, task = Xdf.values.astype(float), ydf.values.astype(float), cfg["task"]
        Z = StandardScaler().fit_transform(X)
        q_main = kaiser_q(Z)
        settings = [("main", q_main)] + ([] if args.no_q2 or q_main == 2 else [("q2", 2)])
        print(f"\n=== {ds}: n={len(X)}, p={X.shape[1]}, Kaiser q={q_main}, feladat={task}")

        # Referencia: eredeti változók (q = p)
        rows.append(evaluate(ds, "Eredeti", X.shape[1], "reference", X, Z, y, task, Z, {}, inductive=False))
        for setting, q in settings:
            for method in config.METHODS:
                H, meta = get_embedding(ds, method, q, Z, args.recompute)
                rows.append(evaluate(ds, method, q, setting, X, Z, y, task, H, meta,
                                     inductive=(setting == "main")))

    res = pd.DataFrame(rows)
    out_path = config.TABLE_DIR / "results_all.csv"
    if out_path.exists() and set(args.datasets) != set(config.DATASETS):
        old = pd.read_csv(out_path)
        res = pd.concat([old[~old.dataset.isin(args.datasets)], res], ignore_index=True)
    res.to_csv(out_path, index=False)
    ranks = add_ranks(res)
    ranks.to_csv(config.TABLE_DIR / "ranks.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3):
        cols = ["dataset", "method", "q", "setting", "max_VIF", "condition_number", "trustworthiness",
                "linear_recon_R2", "cv_R2", "cv_AUC", "cv_R2_inductive", "cv_AUC_inductive",
                "share_sig_coefs", "sign_flip_rate", "fit_time_s"]
        print(res[[c for c in cols if c in res]])
        print(ranks)


if __name__ == "__main__":
    main()
