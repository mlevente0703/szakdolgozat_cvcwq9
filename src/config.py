from pathlib import Path

# --- Elérési utak ---
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# --- Adatbázisok ---
DATASETS = {
    "concrete": {"raw": "Concrete_Data.xlsx"},
    "fat":      {"raw": "FAT_Data.xlsx"},
    "ess":      {"raw": "ESS2020.xlsx"},
}
