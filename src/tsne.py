import sys

import numpy as np
from sklearn.manifold import TSNE

from src import config, data_prep


def fit_tsne(Z: np.ndarray, q: int):
    """t-SNE PCA-alapú kezdőértékekkel, a KL-divergencia minimalizálásával.

    A gyorsító Barnes-Hut-közelítés csak q < 4 esetén működik,
    ezért q >= 4-nél pontos számítás.

    Vissza: (beágyazás n x q, illesztett modell).
    """
    method = "barnes_hut" if q < 4 else "exact"
    model = TSNE(n_components=q, perplexity=config.TSNE_PERPLEXITY,
                 max_iter=config.TSNE_MAX_ITER, init="pca", learning_rate="auto",
                 method=method, random_state=config.RANDOM_STATE, n_jobs=-1)
    H = model.fit_transform(Z)
    return H, model


def main(name: str = "fat") -> None:
    """t-SNE egy adatbázison a Kaiser-féle q-val, a beágyazás mentése.

    Futtatás:  python -m src.tsne fat (vagy: concrete, ess)
    """
    X, _, _ = data_prep.load_dataset(name)   # y és groups itt nem kell
    Z = data_prep.standardize(X)
    q = data_prep.kaiser_q(X)

    H, model = fit_tsne(Z, q)

    print(f"Adatbázis: {name}   X: {X.shape}   Kaiser-q: {q}   H: {H.shape}")
    print(f"Módszer: {model.method}   iterációk: {model.n_iter_} / {config.TSNE_MAX_ITER}")
    print(f"KL-divergencia: {model.kl_divergence_:.3f}")

    path = data_prep.save_embedding(H, name, "t-SNE")
    print(f"Mentve: {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fat")
