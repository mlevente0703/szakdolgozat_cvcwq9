import pandas as pd
import numpy as np

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

# Tisztítás

def clean_concrete() -> pd.DataFrame:
    """Concrete: 1030 sor, 8 magyarázóváltozó + MixID (csoportazonosító)."""
    df = load_raw("concrete")

    df.columns = ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer",
                  "CoarseAggregate", "FineAggregate", "Age", "CompressiveStrength"]

    df = df.drop_duplicates().reset_index(drop=True)

    # Keverékazonosító: azonos összetétel = azonos MixID (428 keverék).
    # Nem magyarázóváltozó, csak a csoportos keresztvalidációhoz kell.

    components = ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer",
                  "CoarseAggregate", "FineAggregate"]
    df["MixID"] = df.groupby(components).ngroup()

    # A kor erősen jobbra ferde -> logaritmus (ferdeség: 3,25 -> -0,14)
    df["Age"] = np.log(df["Age"])
    df = df.rename(columns={"Age": "LogAge"})

    return df


def check_bmi(df: pd.DataFrame, tolerance: float = 0.05) -> pd.DataFrame:
    """A rögzített BMI összevetése a súlyból és magasságból számolttal.

    BMI = 703 * W / H^2  (W: font, H: hüvelyk). Kerekítés miatt kis eltérés
    normális (medián 0,16%), ezért csak a tolerance-nél nagyobb eltérést jelezzük.
    Vissza: az ellentmondásos sorok, a számolt BMI-vel és az eltéréssel együtt.
    """
    bmi_calc = 703 * df["Weight"] / df["Height"] ** 2
    deviation = bmi_calc / df["BMI"] - 1
    inconsistent = deviation.abs() > tolerance
    return df.loc[inconsistent, ["Weight", "Height", "BMI"]].assign(
        BMI_calc=bmi_calc[inconsistent].round(1),
        deviation_pct=(deviation[inconsistent] * 100).round(1),
    )


def clean_fat() -> pd.DataFrame:
    """FAT: 252 sor, 14 magyarázóváltozó."""
    df = load_raw("fat")

    # Ellenőrzés: hol nem egyezik a rögzített BMI a súlyból és magasságból számolttal?
    suspicious = check_bmi(df)

    # Hibás magasság: ahol a BMI ellentmondásos
    # irreális (< 50 hüvelyk = 127 cm).
    # Javítás a rögzített BMI-ből:
    # H = sqrt(703 * W / BMI), negyed hüvelykre kerekítve

    bad = df.index.isin(suspicious.index) & (df["Height"] < 50)
    h = np.sqrt(703 * df.loc[bad, "Weight"] / df.loc[bad, "BMI"])
    df.loc[bad, "Height"] = np.round(h * 4) / 4          # 69,5 hüvelyk

    # 0% testzsír élettanilag lehetetlen -> a sor törlése (182. megfigyelés)
    df = df[df["Fat"] > 0].reset_index(drop=True)

    # Mértékegységek: font -> kg, hüvelyk -> cm (csak az értelmezhetőség miatt)
    df["Weight"] = df["Weight"] * 0.45359237
    df["Height"] = df["Height"] * 2.54
    df = df.rename(columns={"Fat": "BodyFat", "Weight": "Weight_kg", "Height": "Height_cm"})

    return df


def clean_ess() -> pd.DataFrame:
    """ESS2020: 1849 sor, 26 magyarázóváltozó."""
    df = load_raw("ess")
    df = df.rename(columns={"GetVaccince": "GetVaccine"})   # elírás a nyers fájlban

    # Célváltozó: az 'Unknown' sorok törlése, Yes = 1 / No = 0
    df = df[df["GetVaccine"] != "Unknown"].reset_index(drop=True)
    df["GetVaccine"] = (df["GetVaccine"] == "Yes").astype(int)

    # Azonosító és megye törlése (a megye a régióba ágyazott -> tökéletes kollinearitás)
    df = df.drop(columns=["id", "County"])

    # Munkaórák: a 100 feletti (168) érték hibás, de az illető dolgozik
    # -> a dolgozók mediánja (40) 
    # Munkaóra: hiányzó = nem dolgozik -> 0
    # Employed jelző

    hours = "WeeklyWork_Hours"
    workers_median = df.loc[df[hours] <= 100, hours].median()
    df.loc[df[hours] > 100, hours] = workers_median
    df["Employed"] = df[hours].notna().astype(int)
    df[hours] = df[hours].fillna(0)

    # Internethasználat: hiányzó = nem internetezik -> 0 + InternetUser jelző
    net = "NetUsePerDay_Minutes"
    df["InternetUser"] = df[net].notna().astype(int)
    df[net] = df[net].fillna(0)

    # Képzési évek: 25 év felett érvénytelen -> a hiányzókkal együtt mediánnal (12) pótolva
    edu = "Education_Years"
    df.loc[df[edu] > 25, edu] = np.nan
    df[edu] = df[edu].fillna(df[edu].median())

    # Politikai műsor: hiányzó -> medián (30), napi 600 perc felett -> 600
    pol = "PoliticalRadioTVPerDay_Minutes"
    df[pol] = df[pol].fillna(df[pol].median()).clip(upper=600)

    # A percben mért változók erősen ferdék -> ln(1 + x)
    df[pol] = np.log1p(df[pol])
    df[net] = np.log1p(df[net])
    df = df.rename(columns={pol: "LogPoliticalMediaMin", net: "LogNetUseMin"})

    # Dummy-kódolás, a referenciakategória oszlopát elhagyva
    # (különben a dummyk összege 1 lenne -> tökéletes kollinearitás)
    reference = {
        "TrustInParlament": "No",
        "SecretGroupInfluenceWorldPol": "No",
        "ScientistsDecievePublic": "No",
        "COVID19": "No",
        "ContactCOVID19": "No",
        "SomeContactCOVID19": "No",
        "PoliticalPartyPref": "Nem Ismert",
        "Region": "Budapest",
    }
    for col, ref in reference.items():
        dummies = pd.get_dummies(df[col], prefix=col, dtype=int).drop(columns=f"{col}_{ref}")
        df = pd.concat([df.drop(columns=col), dummies], axis=1)

    return df


# A tisztított adatok betöltése a modellekhez

def load_dataset(name: str) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """Tisztított CSV beolvasása és szétválasztása.

    Vissza: X (magyarázóváltozók), y (célváltozó), groups (csak a Concrete-nél
    a MixID a csoportos keresztvalidációhoz, máshol None).
    """
    info = config.DATASETS[name]
    df = pd.read_csv(config.PROCESSED_DIR / info["clean"])

    y = df[info["target"]]
    groups = df[info["groups"]] if info["groups"] else None

    # X = minden más: a célváltozó és a csoportazonosító nem magyarázóváltozó
    drop = [info["target"]] + ([info["groups"]] if info["groups"] else [])
    X = df.drop(columns=drop)

    return X, y, groups


# Diagnosztika

def kaiser_q(X) -> int:
    """Kaiser-kritérium: a korrelációs mátrix 1-nél nagyobb sajátértékeinek száma.

    Legalább 2, hogy minden eljárás ugyanannyi (és ábrázolható számú) dimenziót kapjon.
    """
    corr = np.corrcoef(np.asarray(X, dtype=float), rowvar=False)
    eigenvalues = np.linalg.eigvalsh(corr)
    return max(2, int((eigenvalues > 1).sum()))


def main() -> None:
    """Mindhárom adatbázis tisztítása és mentése: data/processed/*.csv"""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaners = {"concrete": clean_concrete, "fat": clean_fat, "ess": clean_ess}
    for name, clean in cleaners.items():
        df = clean()
        df.to_csv(config.PROCESSED_DIR / config.DATASETS[name]["clean"], index=False)
        print(f"{name:9s} -> {config.DATASETS[name]['clean']}  {df.shape}")


if __name__ == "__main__":
    main()
