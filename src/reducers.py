"""A négy dimenziócsökkentő eljárás egységes felülettel.

Minden reduktor `fit_transform(Z)`-t ad, ahol Z a standardizált magyarázóváltozó-
mátrix (a célváltozó soha nem kerül bele -> mind a négy módszer felügyelet nélküli).

- PCA         : SVD-alapú főkomponens-elemzés (2. fejezet)
- MDS         : metrikus MDS, SMACOF-optimalizálás, több véletlen kezdőkonfiguráció (3. fejezet)
- t-SNE       : perplexitás 30, PCA-inicializálás; q >= 4 esetén 'exact' módszer (4. fejezet)
- Autoencoder : p -> H -> q -> H -> p háló tanh aktivációval, MSE-veszteség, korai leállítás,
                több kezdősúly közül a legkisebb validációs hibájú (5. fejezet)

A PCA és az autoencoder `transform`-ot is tud (új megfigyelésekre alkalmazható),
az MDS és a t-SNE nem (lásd 3.4 és 4.4 fejezet).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import MDS, TSNE
from sklearn.neural_network import MLPRegressor

from src import config


class BaseReducer:
    name = "base"
    supports_transform = False

    def __init__(self, n_components: int, random_state: int = config.RANDOM_STATE):
        self.q = n_components
        self.random_state = random_state
        self.fit_time_ = None
        self.info_: dict = {}

    def fit_transform(self, Z: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        H = self._fit_transform(np.asarray(Z, dtype=float))
        self.fit_time_ = time.perf_counter() - t0
        return H

    def _fit_transform(self, Z):
        raise NotImplementedError

    def transform(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.name}: nincs kifejezett leképezés új megfigyelésekre")


class PCAReducer(BaseReducer):
    name = "PCA"
    supports_transform = True

    def _fit_transform(self, Z):
        self.model = PCA(n_components=self.q, svd_solver="full", random_state=self.random_state)
        T = self.model.fit_transform(Z)
        self.info_ = {
            "explained_variance_ratio": self.model.explained_variance_ratio_.tolist(),
            "cumulative_explained": float(self.model.explained_variance_ratio_.sum()),
            "eigenvalues": self.model.explained_variance_.tolist(),
        }
        return T

    def transform(self, Z):
        return self.model.transform(Z)

    def loadings(self, feature_names) -> "np.ndarray":
        return self.model.components_.T  # p x q sajátvektor-mátrix (V_q)


class MDSReducer(BaseReducer):
    name = "MDS"

    def _fit_transform(self, Z):
        self.model = MDS(n_components=self.q, metric_mds=True, n_init=config.MDS_N_INIT,
                         init="random", max_iter=config.MDS_MAX_ITER, metric="euclidean",
                         normalized_stress=False, random_state=self.random_state, n_jobs=-1)
        Y = self.model.fit_transform(Z)
        self.info_ = {"raw_stress": float(self.model.stress_), "n_iter": int(self.model.n_iter_)}
        return Y


class TSNEReducer(BaseReducer):
    name = "t-SNE"

    def __init__(self, n_components, random_state=config.RANDOM_STATE, perplexity=config.TSNE_PERPLEXITY):
        super().__init__(n_components, random_state)
        self.perplexity = perplexity

    def _fit_transform(self, Z):
        perp = min(self.perplexity, (len(Z) - 1) / 3)  # a perplexitás < n feltétel
        method = "barnes_hut" if self.q < 4 else "exact"  # Barnes-Hut csak q<4-re
        self.model = TSNE(n_components=self.q, perplexity=perp, max_iter=config.TSNE_MAX_ITER,
                          init="pca", learning_rate="auto", method=method,
                          random_state=self.random_state, n_jobs=-1)
        Y = self.model.fit_transform(Z)
        self.info_ = {"kl_divergence": float(self.model.kl_divergence_), "method": method,
                      "perplexity": perp, "n_iter": int(self.model.n_iter_)}
        return Y


class AutoencoderReducer(BaseReducer):
    """Szimmetrikus autoencoder sklearn MLPRegressor-ral (bemenet = kimenet = Z).

    Architektúra: p -> H -> q (bottleneck) -> H -> p, rejtett rétegek tanh aktivációval,
    lineáris kimenet. A látens kód a bottleneck réteg aktivációja:
        h = tanh(W_2 tanh(W_1 x + b_1) + b_2)
    """
    name = "Autoencoder"
    supports_transform = True

    def __init__(self, n_components, random_state=config.RANDOM_STATE,
                 hidden=config.AE_HIDDEN, n_restarts=config.AE_N_RESTARTS):
        super().__init__(n_components, random_state)
        self.hidden = max(hidden, 2 * n_components)
        self.n_restarts = n_restarts

    def _fit_one(self, Z, seed):
        m = MLPRegressor(hidden_layer_sizes=(self.hidden, self.q, self.hidden), activation="tanh",
                         solver="adam", alpha=1e-4, learning_rate_init=1e-3, batch_size=32,
                         max_iter=config.AE_MAX_ITER, early_stopping=True, validation_fraction=0.2,
                         n_iter_no_change=30, tol=1e-5, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m.fit(Z, Z)
        return m

    def _fit_transform(self, Z):
        rng = np.random.RandomState(self.random_state)
        best, scores = None, []
        for _ in range(self.n_restarts):
            m = self._fit_one(Z, int(rng.randint(0, 2**31 - 1)))
            scores.append(m.best_validation_score_)       # validációs R^2
            if best is None or m.best_validation_score_ > best.best_validation_score_:
                best = m
        self.model = best
        recon = self.model.predict(Z)
        self.info_ = {
            "val_r2_best": float(best.best_validation_score_),
            "val_r2_all_restarts": [float(s) for s in scores],
            "recon_mse": float(np.mean((Z - recon) ** 2)),
            "epochs": int(best.n_iter_),
            "architecture": f"{Z.shape[1]}-{self.hidden}-{self.q}-{self.hidden}-{Z.shape[1]}",
        }
        return self.transform(Z)

    def transform(self, Z):
        h = np.asarray(Z, dtype=float)
        for W, b in zip(self.model.coefs_[:2], self.model.intercepts_[:2]):  # csak a kódoló
            h = np.tanh(h @ W + b)
        return h


REDUCERS = {"PCA": PCAReducer, "MDS": MDSReducer, "t-SNE": TSNEReducer, "Autoencoder": AutoencoderReducer}


def make_reducer(method: str, n_components: int, random_state: int = config.RANDOM_STATE) -> BaseReducer:
    return REDUCERS[method](n_components, random_state=random_state)
