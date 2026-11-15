import sys

import numpy as np
from sklearn.neural_network import MLPRegressor

from src import config, data_prep


def fit_autoencoder(Z: np.ndarray, q: int):
    """Szimmetrikus autoencoder: p -> H -> q -> H -> p, H = max(16, 2q).

    A háló a bemenetét tanulja visszaadni (Z -> Z). Rejtett rétegek: tanh,
    kimenet: lineáris, veszteség: négyzetes hiba, Adam + enyhe L2-regularizáció,
    korai leállítás a validációs hányadon. Több kezdősúllyal tanít, és a
    legjobb validációs R^2-ű hálót tartja meg.

    Vissza: (beágyazás n x q, illesztett modell).
    """
    hidden = max(config.AE_HIDDEN, 2 * q)
    best = None
    for i in range(config.AE_N_RESTARTS):
        model = MLPRegressor(hidden_layer_sizes=(hidden, q, hidden), activation="tanh",
                             solver="adam", alpha=1e-4, batch_size=32, tol=1e-5,
                             max_iter=config.AE_MAX_ITER, early_stopping=True,
                             validation_fraction=config.AE_VALIDATION,
                             n_iter_no_change=config.AE_PATIENCE,
                             random_state=config.RANDOM_STATE + i)
        model.fit(Z, Z)
        print(f"  kezdősúly {i + 1}: validációs R^2 = {model.best_validation_score_:.3f}"
              f"   epochok: {model.n_iter_}")
        if best is None or model.best_validation_score_ > best.best_validation_score_:
            best = model
    return encode(best, Z), best


def encode(model: MLPRegressor, Z: np.ndarray) -> np.ndarray:
    """A kódoló fele: h = tanh(W2 · tanh(W1 · z + b1) + b2), a szűk keresztmetszet aktivációja."""
    h = Z
    for W, b in zip(model.coefs_[:2], model.intercepts_[:2]):
        h = np.tanh(h @ W + b)
    return h


def main(name: str = "fat") -> None:
    """Autoencoder egy adatbázison a Kaiser-féle q-val, a beágyazás mentése.

    Futtatás:  python -m src.autoencoder fat (vagy: concrete, ess)
    """
    X, _, _ = data_prep.load_dataset(name)   # y és groups itt nem kell
    Z = data_prep.standardize(X)
    q = data_prep.kaiser_q(X)

    print(f"Adatbázis: {name}   X: {X.shape}   Kaiser-q: {q}")
    H, model = fit_autoencoder(Z, q)

    hidden = model.hidden_layer_sizes[0]
    print(f"Architektúra: {Z.shape[1]}-{hidden}-{q}-{hidden}-{Z.shape[1]}   H: {H.shape}")
    print(f"Legjobb validációs R^2: {model.best_validation_score_:.3f}")

    path = data_prep.save_embedding(H, name, "Autoencoder")
    print(f"Mentve: {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fat")
