import sys
import time

import numpy as np
from sklearn.manifold import MDS

from src import config, data_prep
from src.pca import standardize


def fit_mds(Z: np.ndarray, q: int):
    """Metrikus MDS SMACOF-algoritmussal, euklideszi távolságokon.

    Vissza: (beágyazás n x q, illesztett modell).
    """
    model = MDS(n_components=q, metric_mds=True, metric="euclidean", init="random",
                n_init=config.MDS_N_INIT, max_iter=config.MDS_MAX_ITER,
                normalized_stress=False, random_state=config.RANDOM_STATE, n_jobs=-1)
    H = model.fit_transform(Z)
    return H, model


def main(name: str = "fat") -> None:
    """MDS egy adatbázison a Kaiser-féle q-val, a beágyazás mentése.

    Futtatás:  python -m src.mds fat (vagy: concrete, ess)
    """
    X, _, _ = data_prep.load_dataset(name)   # y és groups itt nem kell
    Z = standardize(X)
    q = data_prep.kaiser_q(X)

    t0 = time.perf_counter()
    H, model = fit_mds(Z, q)
    fit_time = time.perf_counter() - t0

    print(f"Adatbázis: {name}   X: {X.shape}   Kaiser-q: {q}   H: {H.shape}")
    print(f"Illesztési idő: {fit_time:.1f} s   iterációk: {model.n_iter_} / {config.MDS_MAX_ITER}")

    path = data_prep.save_embedding(H, name, "MDS")
    print(f"Mentve: {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fat")
