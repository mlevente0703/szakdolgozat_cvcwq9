import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression

from src import config, data_prep

pd.set_option("display.width", 200)


# Multikollinearitás

def collinearity(A: np.ndarray) -> dict:
    """VIF és kondíciószám a változók korrelációs mátrixából.

    VIF_j = 1 / (1 - R_j^2) = a korrelációs mátrix inverzének j-edik főátlóbeli eleme.
    Kondíciószám: a legnagyobb és a legkisebb sajátérték hányadosa.
    """
    corr = np.corrcoef(A, rowvar=False)
    vif = np.diag(np.linalg.inv(corr))
    eigenvalues = np.linalg.eigvalsh(corr)
    return {"max_VIF": vif.max(), "mean_VIF": vif.mean(),
            "condition_number": eigenvalues.max() / eigenvalues.min()}


# Magyarázóerő

def fit_model(A: np.ndarray, y: pd.Series, task: str):
    """A célváltozó modellje konstanssal: regressziónál OLS, osztályozásnál logit."""
    exog = sm.add_constant(A)
    if task == "regression":
        return sm.OLS(y, exog).fit()
    return sm.Logit(y, exog).fit(disp=0)


def explanatory_power(fit, task: str, k: int) -> dict:
    """R^2 és korrigált R^2 (logitnál McFadden-féle pszeudo-R^2), k: magyarázóváltozók száma.

    A szignifikáns arány a konstans nélküli együtthatókra vonatkozik (p < 0,05).
    """
    if task == "regression":
        r2, r2_adj = fit.rsquared, fit.rsquared_adj
    else:
        r2 = fit.prsquared                               # 1 - lnL / lnL_0
        r2_adj = 1 - (fit.llf - k) / fit.llnull          # korrigált McFadden
    return {"R2": r2, "R2_adj": r2_adj, "sig_share": (fit.pvalues[1:] < 0.05).mean()}


# Értelmezhetőség

def interpretability_r2(Z: np.ndarray, H: np.ndarray) -> float:
    """Az új tengelyek mennyire írhatók fel az eredeti változók lineáris kombinációjaként.

    Minden H-oszlopot Z-re regresszálunk, az R^2-ek átlaga. PCA-nál pontosan 1.
    """
    return LinearRegression().fit(Z, H).score(Z, H)


# A módszerek hasonlósága

def distance_correlations(embeddings: dict) -> pd.DataFrame:
    """A páronkénti távolságok Spearman-korrelációja minden módszerpárra.

    A tengelyek önkényesek (forgatás, tükrözés, skála), ezért a távolságokat vetjük össze.
    Spearman = a rangsorolt távolságok Pearson-korrelációja.
    """
    ranks = {m: rankdata(pdist(H)) for m, H in embeddings.items()}
    names = list(ranks)
    corr = np.corrcoef([ranks[m] for m in names])
    return pd.DataFrame(corr, index=names, columns=names)


# Összesítés

def evaluate(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Egy adatbázis: az Eredeti (standardizált X) és a négy beágyazás összevetése."""
    X, y, _ = data_prep.load_dataset(name)
    task = config.DATASETS[name]["task"]
    Z = data_prep.standardize(X)
    q = data_prep.kaiser_q(X)

    embeddings = {"Eredeti": Z}
    for method in config.METHODS:
        embeddings[method] = data_prep.load_embedding(name, method, q)

    rows = []
    for method, H in embeddings.items():
        k = H.shape[1]
        fit = fit_model(H, y, task)
        rows.append({"dataset": name, "method": method, "q": k,
                     **explanatory_power(fit, task, k),
                     **collinearity(H),
                     "interpretability_R2": interpretability_r2(Z, H)})

    table = pd.DataFrame(rows)
    # az Eredeti modell magyarázóerejének hányad részét őrzi meg a módszer
    table.insert(5, "R2_retained", table["R2"] / table["R2"].iloc[0])
    return table, distance_correlations(embeddings)


def main() -> None:
    """Mindhárom adatbázis összehasonlítása, mentés: results/tables/

    Futtatás:  python -m src.evaluation
    """
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tables = []
    for name in config.DATASETS:
        table, dist_corr = evaluate(name)
        tables.append(table)
        dist_corr.to_csv(config.TABLE_DIR / f"distance_corr_{name}.csv")
        print(f"\n{name.upper()}  ({config.DATASETS[name]['task']})")
        print(table.drop(columns="dataset").round(3).to_string(index=False))
        print("\nTávolságmátrixok Spearman-korrelációja:")
        print(dist_corr.round(2))

    pd.concat(tables).to_csv(config.TABLE_DIR / "comparison.csv", index=False)
    print(f"\nMentve: {config.TABLE_DIR.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
