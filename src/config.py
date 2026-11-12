from pathlib import Path

# --- Elérési utak ---
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# --- Adatbázisok ---
DATASETS = {
    "concrete": {"raw": "Concrete_Data.xlsx", "clean": "concrete_clean.csv"},
    "fat":      {"raw": "FAT_Data.xlsx", "clean": "fat_clean.csv"},
    "ess":      {"raw": "ESS2020.xlsx", "clean": "ess_clean.csv"},
}
