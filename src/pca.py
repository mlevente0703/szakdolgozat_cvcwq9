import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src import config, data_prep


def standardize(X) -> np.ndarray:
    """Nulla átlag, egységnyi szórás minden oszlopban.
    Így a mértékegységek nem számítanak.
    """
    return StandardScaler().fit_transform(X)


def fit_pca(Z: np.ndarray, q: int):
    """A Z korrelációs mátrixának q legnagyobb sajátértékéhez tartozó sajátvektorokra
    vetít (szinguláris felbontással számolva). A főkomponensek korrelálatlanok.

    Vissza: (beágyazás n x q, illesztett modell). A modell .transform()-ja kell
    majd az induktív keresztvalidációhoz (új megfigyelések leképezése).
    """
    model = PCA(n_components=q, svd_solver="full")
    H = model.fit_transform(Z)
    return H, model


def main(name: str = "fat") -> None:
    """Gyors ellenőrzés egy adatbázison: PCA a Kaiser-féle q-val.

    Futtatás:  python -m src.pca fat (vagy: concrete, ess)
    """
    X, _, _ = data_prep.load_dataset(name)   # y és groups itt nem kell
    Z = standardize(X)
    q = data_prep.kaiser_q(X)
    H, model = fit_pca(Z, q)

    print(f"Adatbázis: {name}   X: {X.shape}   Kaiser-q: {q}   H: {H.shape}")
    print("\nMagyarázott variancia főkomponensenként:")
    for i, ratio in enumerate(model.explained_variance_ratio_, start=1):
        print(f"  PC{i}: {ratio:6.1%}")
    print(f"  Összesen: {model.explained_variance_ratio_.sum():.1%}")

    corr = np.corrcoef(H, rowvar=False)
    max_r = np.abs(corr - np.eye(q)).max()
    print(f"\nLegnagyobb |r| a főkomponensek között: {max_r:.1e}  (0 = korrelálatlanok)")

    path = data_prep.save_embedding(H, name, "PCA")
    print(f"Mentve: {path.relative_to(config.ROOT)}")



if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fat")
