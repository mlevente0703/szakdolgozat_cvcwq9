"""Multikollinearitás-diagnosztika (szakdolgozat 1.4 fejezet).

- páronkénti korrelációs mátrix
- varianciainflációs tényező: VIF_j = 1 / (1 - R_j^2)
- a korrelációs mátrix sajátértékei és a kondíciószám (lambda_max / lambda_min)

Ugyanezeket a mutatókat számoljuk a dimenziócsökkentés UTÁNI változókra is,
így közvetlenül mérhető, mennyire szüntette meg egy módszer a multikollinearitást.

Futtatás (a tisztított adatokra):  python -m src.diagnostics
"""
import numpy as np
import pandas as pd

from src.config import DATASETS, PROCESSED_DIR, TABLE_DIR


def vif(X: np.ndarray) -> np.ndarray:
    """VIF minden oszlopra: a korrelációs mátrix inverzének főátlója.

    Standardizált változók esetén VIF_j = [R^{-1}]_jj, ami megegyezik az
    1/(1-R_j^2) definícióval, de egyetlen mátrixinverzióval számolható.
    Közel szinguláris esetben pszeudoinverzt használunk.
    """
    X = np.asarray(X, dtype=float)
    if X.shape[1] < 2:
        return np.ones(X.shape[1])
    R = np.corrcoef(X, rowvar=False)
    try:
        inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(R)
    return np.diag(inv)


def condition_number(X: np.ndarray) -> float:
    """Kondíciószám a korrelációs mátrix sajátértékeiből: lambda_max / lambda_min."""
    X = np.asarray(X, dtype=float)
    if X.shape[1] < 2:
        return 1.0
    ev = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))
    return float(ev.max() / max(ev.min(), 1e-12))


def collinearity_summary(X: np.ndarray) -> dict:
    """Összefoglaló mutatók egy magyarázóváltozó-mátrixra."""
    X = np.asarray(X, dtype=float)
    R = np.corrcoef(X, rowvar=False)
    off = np.abs(R[~np.eye(R.shape[0], dtype=bool)]) if R.ndim == 2 and R.shape[0] > 1 else np.array([0.0])
    v = vif(X)
    return {
        "n_features": X.shape[1],
        "max_abs_corr": float(off.max()),
        "mean_abs_corr": float(off.mean()),
        "max_VIF": float(v.max()),
        "mean_VIF": float(v.mean()),
        "n_VIF_gt_5": int((v > 5).sum()),
        "n_VIF_gt_10": int((v > 10).sum()),
        "condition_number": condition_number(X),
    }


def load_dataset(name: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    cfg = DATASETS[name]
    df = pd.read_csv(PROCESSED_DIR / cfg["file"])
    y = df[cfg["target"]]
    X = df.drop(columns=[cfg["target"]])
    return X, y, cfg


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in DATASETS:
        X, _, _ = load_dataset(name)
        s = collinearity_summary(X.values)
        ev = np.sort(np.linalg.eigvalsh(np.corrcoef(X.values, rowvar=False)))[::-1]
        s.update({"dataset": name, "n_obs": len(X), "kaiser_q": int((ev > 1).sum())})
        rows.append(s)
        pd.DataFrame({"variable": X.columns, "VIF": vif(X.values)}).sort_values("VIF", ascending=False) \
            .to_csv(TABLE_DIR / f"vif_original_{name}.csv", index=False)
        X.corr().round(3).to_csv(TABLE_DIR / f"corr_original_{name}.csv")
        pd.DataFrame({"component": np.arange(1, len(ev) + 1), "eigenvalue": ev,
                      "explained_ratio": ev / ev.sum(), "cumulative": np.cumsum(ev) / ev.sum()}) \
            .to_csv(TABLE_DIR / f"eigenvalues_{name}.csv", index=False)
    out = pd.DataFrame(rows).set_index("dataset")
    out.to_csv(TABLE_DIR / "collinearity_original.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.round(2))


if __name__ == "__main__":
    main()
