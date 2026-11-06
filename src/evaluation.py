"""Értékelési mutatók a dimenziócsökkentő módszerek összehasonlításához.

Öt mutatócsoport:
  A) Multikollinearitás az új változókon  (1.4 fejezet: VIF, kondíciószám, korreláció)
  B) Szerkezetmegőrzés                    (trustworthiness, Shepard-korreláció, Stress-1,
                                           az eredeti változók lineárisan visszaállítható varianciája)
  C) Előrejelző teljesítmény              (ismételt k-szoros keresztvalidáció)
  D) Becslési stabilitás                  (1.3 fejezet: szignifikáns együtthatók, bootstrap
                                           előjelváltás és együttható-szóródás)
  E) Futási idő

Módszertani megjegyzés a C) csoporthoz: az MDS és a t-SNE nem ad leképezést új
megfigyelésekre, ezért a "transzduktív" protokollban a beágyazást a TELJES X-en
számoljuk (a célváltozó nélkül, tehát y-szivárgás nincs), és csak a ráépülő regressziót
keresztvalidáljuk. Ez mind a négy módszerre azonos, így összehasonlítható. A PCA-ra és az
autoencoderre emellett "induktív" CV-t is számolunk, ahol a reduktor is csak a tanítóhajtáson
tanul - ez a valódi predikciós helyzetet modellezi.
"""
from __future__ import annotations

import warnings

import numpy as np
import statsmodels.api as sm
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.metrics import (accuracy_score, log_loss, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src import config
from src.diagnostics import collinearity_summary

N_BOOT = 200
TRUST_K = 10

# Melyik irány a jobb az egyes mutatóknál (rangsoroláshoz). +1: nagyobb jobb, -1: kisebb jobb.
METRIC_DIRECTION = {
    "max_VIF": -1, "mean_VIF": -1, "condition_number": -1, "max_abs_corr": -1,
    "trustworthiness": +1, "shepard_spearman": +1, "stress1": -1, "linear_recon_R2": +1,
    "cv_R2": +1, "cv_RMSE": -1, "cv_MAE": -1,
    "cv_AUC": +1, "cv_accuracy": +1, "cv_logloss": -1,
    "share_sig_coefs": +1, "sign_flip_rate": -1, "boot_coef_sd": -1,
    "fit_time_s": -1,
}


# ---------------------------------------------------------------- A) multikollinearitás
def collinearity_metrics(H: np.ndarray) -> dict:
    s = collinearity_summary(H)
    return {k: s[k] for k in ["max_VIF", "mean_VIF", "condition_number", "max_abs_corr"]}


# ---------------------------------------------------------------- B) szerkezetmegőrzés
def structure_metrics(Z: np.ndarray, H: np.ndarray, random_state=config.RANDOM_STATE) -> dict:
    # Távolságalapú mutatók: nagy n esetén véletlen részmintán (a pdist n^2/2 párt ad)
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(Z), size=min(len(Z), 1500), replace=False)
    delta, d = pdist(Z[idx]), pdist(H[idx])
    # Stress-1 optimális skálázás után (a t-SNE / AE koordináták skálája önkényes)
    b = (delta @ d) / (d @ d)
    stress1 = float(np.sqrt(np.sum((delta - b * d) ** 2) / np.sum((b * d) ** 2)))
    # Az eredeti (standardizált) változók varianciájának hányada, amely a beágyazásból
    # lineárisan visszaállítható. PCA esetén ez pontosan a kumulált magyarázott variancia.
    Zhat = LinearRegression().fit(H, Z).predict(H)
    recon_r2 = 1 - np.sum((Z - Zhat) ** 2) / np.sum((Z - Z.mean(0)) ** 2)
    return {
        "trustworthiness": float(trustworthiness(Z, H, n_neighbors=TRUST_K)),
        "shepard_spearman": float(spearmanr(delta, d).statistic),
        "stress1": stress1,
        "linear_recon_R2": float(recon_r2),
    }


# ---------------------------------------------------------------- C) előrejelzés
def _model(task):
    if task == "regression":
        return LinearRegression()
    return LogisticRegression(penalty=None, max_iter=5000)


def _cv_splitter(task, repeats, seed):
    if task == "regression":
        return RepeatedKFold(n_splits=config.CV_FOLDS, n_repeats=repeats, random_state=seed)
    return RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=repeats, random_state=seed)


def _fold_scores(task, y_true, pred):
    if task == "regression":
        return {"cv_R2": r2_score(y_true, pred),
                "cv_RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
                "cv_MAE": mean_absolute_error(y_true, pred)}
    return {"cv_AUC": roc_auc_score(y_true, pred),
            "cv_accuracy": accuracy_score(y_true, pred >= 0.5),
            "cv_logloss": log_loss(y_true, pred)}


def _predict(model, task, X):
    return model.predict(X) if task == "regression" else model.predict_proba(X)[:, 1]


def cv_transductive(H: np.ndarray, y: np.ndarray, task: str, seed=config.RANDOM_STATE) -> dict:
    """Ismételt k-szoros CV egy rögzített (felügyelet nélkül előállított) beágyazáson."""
    scores = []
    for tr, te in _cv_splitter(task, config.CV_REPEATS, seed).split(H, y):
        sc = StandardScaler().fit(H[tr])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = _model(task).fit(sc.transform(H[tr]), y[tr])
        scores.append(_fold_scores(task, y[te], _predict(m, task, sc.transform(H[te]))))
    keys = scores[0].keys()
    out = {k: float(np.mean([s[k] for s in scores])) for k in keys}
    out.update({f"{k}_sd": float(np.std([s[k] for s in scores])) for k in keys})
    return out


def cv_inductive(X: np.ndarray, y: np.ndarray, task: str, reducer_factory, seed=config.RANDOM_STATE) -> dict:
    """k-szoros CV, ahol a standardizálás ÉS a dimenziócsökkentés is csak a tanítóhajtáson tanul.
    Csak `transform`-mal rendelkező módszerekre (PCA, autoencoder)."""
    splitter = (KFold(config.CV_FOLDS, shuffle=True, random_state=seed) if task == "regression"
                else StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=seed))
    scores = []
    for tr, te in splitter.split(X, y):
        sx = StandardScaler().fit(X[tr])
        red = reducer_factory()
        Htr = red.fit_transform(sx.transform(X[tr]))
        Hte = red.transform(sx.transform(X[te]))
        sh = StandardScaler().fit(Htr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = _model(task).fit(sh.transform(Htr), y[tr])
        scores.append(_fold_scores(task, y[te], _predict(m, task, sh.transform(Hte))))
    return {f"{k}_inductive": float(np.mean([s[k] for s in scores])) for k in scores[0]}


# ---------------------------------------------------------------- D) becslési stabilitás
def _fit_coefs(task, H, y):
    if task == "regression":
        Hc = np.column_stack([np.ones(len(H)), H])
        return np.linalg.lstsq(Hc, y, rcond=None)[0][1:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticRegression(penalty=None, max_iter=5000).fit(H, y).coef_.ravel()


def inference_metrics(H: np.ndarray, y: np.ndarray, task: str, seed=config.RANDOM_STATE) -> dict:
    """Teljes mintás modell t-/z-próbái és bootstrap együttható-stabilitás.

    A magyarázóváltozókat (és regressziónál y-t is) standardizáljuk, így az együtthatók
    és szórásaik a módszerek között összehasonlíthatók.
    """
    Hs = StandardScaler().fit_transform(H)
    ys = (y - y.mean()) / y.std() if task == "regression" else y
    Hc = sm.add_constant(Hs, has_constant="add")
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if task == "regression":
                res = sm.OLS(ys, Hc).fit()
                out.update({"insample_R2": res.rsquared, "insample_adjR2": res.rsquared_adj,
                            "F_pvalue": res.f_pvalue})
            else:
                res = sm.Logit(ys, Hc).fit(disp=0, maxiter=500)
                out.update({"pseudo_R2": res.prsquared, "LLR_pvalue": res.llr_pvalue})
            pv = np.asarray(res.pvalues)[1:]
            out["share_sig_coefs"] = float(np.mean(pv < 0.05))
            out["n_sig_coefs"] = int(np.sum(pv < 0.05))
        except Exception as e:  # pl. tökéletes szeparáció logitnál
            out["inference_error"] = str(e)[:80]

    # Bootstrap: hányszor fordul meg az együttható előjele a teljes mintás becsléshez képest,
    # és mekkora a standardizált együtthatók szóródása (1.3 fejezet: instabil becslések).
    base = _fit_coefs(task, Hs, ys)
    rng = np.random.RandomState(seed)
    boots = np.empty((N_BOOT, Hs.shape[1]))
    for b in range(N_BOOT):
        i = rng.randint(0, len(Hs), len(Hs))
        boots[b] = _fit_coefs(task, Hs[i], ys[i])
    out["sign_flip_rate"] = float(np.mean(np.sign(boots) != np.sign(base)))
    out["boot_coef_sd"] = float(np.mean(boots.std(axis=0)))
    return out
