"""Központi beállítások: elérési utak, véletlenmag, adatbázis-specifikus paraméterek."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

RANDOM_STATE = 42
CV_FOLDS = 5          # keresztvalidáció hajtásszáma
CV_REPEATS = 3        # ismételt keresztvalidáció

# Adatbázisok: célváltozó és feladattípus.
# task = "regression" -> OLS, task = "classification" -> logisztikus regresszió
DATASETS = {
    "concrete": {"file": "concrete_clean.csv", "target": "CompressiveStrength", "task": "regression"},
    "fat":      {"file": "fat_clean.csv",      "target": "BodyFat",             "task": "regression"},
    "ess":      {"file": "ess_clean.csv",      "target": "GetVaccine",          "task": "classification"},
}

# Dimenziócsökkentő módszerek
METHODS = ["PCA", "MDS", "t-SNE", "Autoencoder"]

# t-SNE alapbeállítások (van der Maaten & Hinton, 2008)
TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 1000

# MDS: több kezdőkonfiguráció, a legkisebb stresszű megoldás marad
MDS_N_INIT = 4
MDS_MAX_ITER = 300

# Autoencoder: több kezdősúllyal tanítjuk, a legkisebb validációs hibájú marad
AE_HIDDEN = 16
AE_N_RESTARTS = 5
AE_MAX_ITER = 2000
