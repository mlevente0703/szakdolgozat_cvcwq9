from pathlib import Path

# --- Elérési utak ---
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
EMBEDDING_DIR = ROOT / "results" / "embeddings"   # a módszerek kimenete (n x q koordináták)

# --- Reprodukálhatóság ---
RANDOM_STATE = 42

# --- Adatbázisok ---
# groups: csoportazonosító oszlop a csoportos keresztvalidációhoz (nem magyarázóváltozó)
DATASETS = {
    "concrete": {"raw": "Concrete_Data.xlsx", "clean": "concrete_clean.csv",
                 "target": "CompressiveStrength", "groups": "MixID"},
    "fat":      {"raw": "FAT_Data.xlsx", "clean": "fat_clean.csv",
                 "target": "BodyFat", "groups": None},
    "ess":      {"raw": "ESS2020.xlsx", "clean": "ess_clean.csv",
                 "target": "GetVaccine", "groups": None},
}

# --- MDS ---
MDS_N_INIT = 4          # véletlen kezdőkonfigurációk száma, a legkisebb stresszű marad
MDS_MAX_ITER = 300      # SMACOF-iterációk felső korlátja

# --- t-SNE ---
TSNE_PERPLEXITY = 30    # kb. ennyi szomszédot vesz figyelembe egy megfigyelés
TSNE_MAX_ITER = 1000    # gradiens-iterációk felső korlátja

# --- Autoencoder ---
AE_HIDDEN = 16          # rejtett réteg mérete: H = max(AE_HIDDEN, 2q)
AE_N_RESTARTS = 5       # kezdősúlyok száma, a legjobb validációs eredményű marad
AE_MAX_ITER = 2000      # epochok felső korlátja
AE_VALIDATION = 0.2     # validációs hányad a korai leállításhoz
AE_PATIENCE = 30        # ennyi epoch javulás nélkül -> leállás
