"""Adattisztítás a három nyers Excel-állományra.

Alapelv: a tisztítás csak a hibás / hiányzó / technikai problémákat kezeli, a
magyarázóváltozók közötti multikollinearitást SZÁNDÉKOSAN megtartja, mert a
dimenziócsökkentő módszereket éppen ennek kezelésén hasonlítjuk össze.
Egyetlen kivétel a tökéletes (egzakt) kollinearitás, mert ott az X^T X mátrix
nem invertálható, és az OLS-referenciamodell sem becsülhető.

Futtatás:  python -m src.data_cleaning
"""
import unicodedata

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, TABLE_DIR

LBS_TO_KG = 0.45359237
INCH_TO_CM = 2.54

log: list[dict] = []


def _log(dataset: str, step: str, detail: str, n_rows: int) -> None:
    log.append({"dataset": dataset, "step": step, "detail": detail, "n_rows_after": n_rows})
    print(f"[{dataset}] {step}: {detail} (n={n_rows})")


# ---------------------------------------------------------------------------
# 1) Concrete Compressive Strength (Yeh, 1998; UCI)
# ---------------------------------------------------------------------------
def clean_concrete() -> pd.DataFrame:
    df = pd.read_excel(RAW_DIR / "Concrete_Data.xlsx", sheet_name="Original Data")
    df.columns = ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer",
                  "CoarseAggregate", "FineAggregate", "Age", "CompressiveStrength"]
    _log("concrete", "beolvasás", "'Original Data' munkalap, 8 magyarázó + 1 célváltozó", len(df))

    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    _log("concrete", "duplikátumok", f"{n0 - len(df)} teljesen azonos sor törölve", len(df))

    _log("concrete", "hiányzó értékek", f"{int(df.isna().sum().sum())} db, nincs teendő", len(df))

    # A kor (nap) erősen jobbra ferde (1-365 nap, néhány diszkrét érték), a
    # szilárdság a korral logaritmikusan nő -> log-transzformáció.
    df["Age"] = np.log(df["Age"])
    df = df.rename(columns={"Age": "LogAge"})
    _log("concrete", "transzformáció", "Age -> ln(Age) a ferdeség miatt", len(df))

    # A multikollinearitás itt szerkezeti: a 7 komponens összege közel állandó
    # (~2340 kg/m^3), ezért a komponensek egymás rovására változnak. Megtartjuk.
    return df


# ---------------------------------------------------------------------------
# 2) Body Fat (Johnson, 1996)
# ---------------------------------------------------------------------------
def clean_fat() -> pd.DataFrame:
    df = pd.read_excel(RAW_DIR / "FAT_Data.xlsx")
    _log("fat", "beolvasás", "14 magyarázó + 1 célváltozó (Fat)", len(df))

    # Ismert adathiba: 42. megfigyelés (0-s index: 41) magassága 29,5 hüvelyk.
    # A rögzített BMI-ből visszaszámolható: H = sqrt(703 * W / BMI) ~ 69,4 hüvelyk.
    bad_h = df["Height"] < 50
    for i in df.index[bad_h]:
        new_h = round(float(np.sqrt(703 * df.at[i, "Weight"] / df.at[i, "BMI"])) * 4) / 4
        _log("fat", "hibás magasság javítása",
             f"sor {i}: {df.at[i, 'Height']} -> {new_h} hüvelyk (BMI-ből visszaszámolva)", len(df))
        df.at[i, "Height"] = new_h

    # 0% testzsír élettanilag lehetetlen -> törlés.
    n0 = len(df)
    df = df[df["Fat"] > 0].reset_index(drop=True)
    _log("fat", "lehetetlen célérték", f"{n0 - len(df)} sor törölve (Fat = 0%)", len(df))

    # Mértékegység-átváltás (lbs -> kg, hüvelyk -> cm) az értelmezhetőség miatt.
    # A kerületek már cm-ben vannak. A standardizálás miatt ez az eredményt nem befolyásolja.
    df["Weight"] = df["Weight"] * LBS_TO_KG
    df["Height"] = df["Height"] * INCH_TO_CM
    df = df.rename(columns={"Fat": "BodyFat", "Weight": "Weight_kg", "Height": "Height_cm"})
    _log("fat", "mértékegység", "Weight lbs->kg, Height inch->cm", len(df))

    # Megjegyzés: a 39. sor (Weight ~165 kg) extrém, de valós megfigyelés -> megtartjuk.
    # A BMI a súlyból és magasságból származtatott változó -> szerkezeti
    # multikollinearitás, szándékosan megtartjuk.
    cols = ["BodyFat"] + [c for c in df.columns if c != "BodyFat"]
    return df[cols]


# ---------------------------------------------------------------------------
# 3) European Social Survey 2020, Magyarország
# ---------------------------------------------------------------------------
YES_NO_UNK = ["TrustInParlament", "SecretGroupInfluenceWorldPol", "ScientistsDecievePublic",
              "COVID19", "ContactCOVID19", "SomeContactCOVID19"]


def clean_ess() -> pd.DataFrame:
    df = pd.read_excel(RAW_DIR / "ESS2020.xlsx")
    df = df.rename(columns={"GetVaccince": "GetVaccine"})
    _log("ess", "beolvasás", f"{df.shape[1]} oszlop", len(df))

    # Célváltozó: hajlandó-e beoltatni magát (Igen=1 / Nem=0). 'Unknown' -> törlés.
    n0 = len(df)
    df = df[df["GetVaccine"].isin(["Yes", "No"])].copy()
    df["GetVaccine"] = (df["GetVaccine"] == "Yes").astype(int)
    _log("ess", "célváltozó", f"{n0 - len(df)} 'Unknown' válasz törölve, Yes=1/No=0", len(df))

    # Azonosító nem magyarázóváltozó.
    df = df.drop(columns=["id"])

    # A megye a régióba ágyazott: dummy-kódolva a megye-dummyk összege pontosan
    # kiadja a régió-dummykat -> TÖKÉLETES kollinearitás. A régiót tartjuk meg.
    df = df.drop(columns=["County"])
    _log("ess", "tökéletes kollinearitás", "County törölve (Region-be ágyazott)", len(df))

    # --- Numerikus változók ---
    # Képzési évek: 25 év felett irreális -> hiányzónak tekintjük.
    n_bad = int((df["Education_Years"] > 25).sum())
    df.loc[df["Education_Years"] > 25, "Education_Years"] = np.nan
    med = df["Education_Years"].median()
    n_na = int(df["Education_Years"].isna().sum())
    df["Education_Years"] = df["Education_Years"].fillna(med)
    _log("ess", "Education_Years", f"{n_bad} db >25 év érvénytelen; {n_na} hiányzó medián ({med})-nal pótolva", len(df))

    # Heti munkaórák: hiányzó = nem dolgozik (az ESS csak a dolgozóktól kérdezi).
    df["Employed"] = df["WeeklyWork_Hours"].notna().astype(int)
    n_cap = int((df["WeeklyWork_Hours"] > 100).sum())
    df["WeeklyWork_Hours"] = df["WeeklyWork_Hours"].clip(upper=100).fillna(0)
    _log("ess", "WeeklyWork_Hours", f"hiányzó -> 0 + Employed indikátor; {n_cap} db >100 óra 100-ra vágva", len(df))

    # Napi internethasználat: hiányzó = nem internetezik (az ESS csak a felhasználóktól kérdezi).
    df["InternetUser"] = df["NetUsePerDay_Minutes"].notna().astype(int)
    df["NetUsePerDay_Minutes"] = df["NetUsePerDay_Minutes"].fillna(0)
    _log("ess", "NetUsePerDay_Minutes", "hiányzó -> 0 + InternetUser indikátor", len(df))

    # Politikai hírfogyasztás: kevés hiányzó -> medián.
    n_na = int(df["PoliticalRadioTVPerDay_Minutes"].isna().sum())
    med = df["PoliticalRadioTVPerDay_Minutes"].median()
    df["PoliticalRadioTVPerDay_Minutes"] = df["PoliticalRadioTVPerDay_Minutes"].fillna(med)
    _log("ess", "PoliticalRadioTVPerDay_Minutes", f"{n_na} hiányzó medián ({med})-nal pótolva", len(df))

    # A percben mért változók erősen jobbra ferdék -> log(1+x).
    for c in ["PoliticalRadioTVPerDay_Minutes", "NetUsePerDay_Minutes"]:
        df[c] = np.log1p(df[c])
    df = df.rename(columns={"PoliticalRadioTVPerDay_Minutes": "LogPoliticalMediaMin",
                            "NetUsePerDay_Minutes": "LogNetUseMin"})
    _log("ess", "transzformáció", "percben mért változók -> ln(1+x)", len(df))

    # --- Kategorikus változók: dummy-kódolás explicit referenciakategóriával ---
    dummies = []
    for c in YES_NO_UNK:
        for lvl in ["Yes", "Unknown"]:           # referencia: "No"
            if (df[c] == lvl).any():
                dummies.append(pd.Series((df[c] == lvl).astype(int), name=f"{c}_{lvl}"))
    party_map = {"Fidesz-KDNP": "Fidesz", "Egyesült Ellenzék": "Ellenzek", "Egyéb": "Egyeb"}
    for lvl, short in party_map.items():           # referencia: "Nem Ismert"
        dummies.append(pd.Series((df["PoliticalPartyPref"] == lvl).astype(int), name=f"Party_{short}"))
    for lvl in sorted(df["Region"].unique()):      # referencia: Budapest
        if lvl != "Budapest":
            ascii_lvl = unicodedata.normalize("NFKD", lvl).encode("ascii", "ignore").decode()
            name = "Region_" + ascii_lvl.replace("-", "")
            dummies.append(pd.Series((df["Region"] == lvl).astype(int), name=name))
    df = pd.concat([df.drop(columns=YES_NO_UNK + ["PoliticalPartyPref", "Region"]),
                    pd.concat(dummies, axis=1)], axis=1)
    _log("ess", "dummy-kódolás", "Yes/Unknown (ref: No), párt (ref: Nem Ismert), régió (ref: Budapest)", len(df))

    # Megjegyzés: SomeContactCOVID19 = COVID19 VAGY ContactCOVID19 (logikai függvény),
    # valamint az 'Unknown' válaszok ugyanazoknál a válaszadóknál halmozódnak ->
    # mérési/szerkezeti multikollinearitás, szándékosan megtartjuk.
    df = df.reset_index(drop=True)
    cols = ["GetVaccine"] + [c for c in df.columns if c != "GetVaccine"]
    return df[cols]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {"concrete_clean.csv": clean_concrete(),
               "fat_clean.csv": clean_fat(),
               "ess_clean.csv": clean_ess()}
    for name, df in outputs.items():
        df.to_csv(PROCESSED_DIR / name, index=False)
        print(f"-> {name}: {df.shape}")
    pd.DataFrame(log).to_csv(TABLE_DIR / "cleaning_log.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
