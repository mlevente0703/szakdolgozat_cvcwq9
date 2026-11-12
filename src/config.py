from pathlib import Path

# --- Elérési utak ---
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

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
