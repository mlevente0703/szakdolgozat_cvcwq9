import pandas as pd

from src import config

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def load_raw(name: str) -> pd.DataFrame:
    path = config.RAW_DIR / config.DATASETS[name]["raw"]
    return pd.read_excel(path)


def show_overview(name: str, df: pd.DataFrame) -> None:
    """Az adatbázis szerkezetének kiírása és az első 5 sor."""
    print("=" * 80)
    print(f"{name.upper()}  ({config.DATASETS[name]['raw']})")
    print("=" * 80)
    df.info()
    print()
    print(df.head())
    print()


def show_details(df: pd.DataFrame) -> None:
    """Hibás értékek felderítése: leíró statisztikák, kategóriák, duplikált sorok."""
    # Numerikus oszlopok: min / max / átlag stb.
    print(df.describe().T.round(2))
    print()

    # Szöveges (kategorikus) oszlopok
    for col in df.select_dtypes(exclude="number").columns:
        print(df[col].value_counts().to_dict())
        print(f"  ^ {col}\n")

    print(f"Duplikált sorok száma: {df.duplicated().sum()}")
    print()


def main() -> None:
    for name in config.DATASETS:
        df = load_raw(name)
        show_overview(name, df)
        show_details(df)


if __name__ == "__main__":
    main()
