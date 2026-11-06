# Dimenziócsökkentő algoritmusok empirikus hatékonyságvizsgálata multikollinearitás kezelésében

Szakdolgozathoz tartozó kód: négy dimenziócsökkentő eljárás (PCA, MDS, t-SNE, autoencoder)
összehasonlítása három, multikollineáris magyarázóváltozókat tartalmazó adatbázison
(4 × 3 = 12 modell).

| Adatbázis | n (tisztítás után) | p | Célváltozó | Feladat | Multikollinearitás eredete |
|---|---|---|---|---|---|
| Concrete (UCI) | 1005 | 8 | nyomószilárdság (MPa) | regresszió (OLS) | szerkezeti (keverékarányok összege közel állandó) |
| FAT (testzsír) | 251 | 14 | testzsír-százalék | regresszió (OLS) | természetes (testméretek) + szerkezeti (BMI) |
| ESS2020 (HU) | 1766 | 26 | beoltatná-e magát (0/1) | osztályozás (logit) | mérési / szerkezeti (átfedő COVID-kérdések, „Unknown” válaszok, indikátorok) |

## Futtatás

```bash
pip install -r requirements.txt
python -m src.data_cleaning      # 1. nyers Excel -> data/processed/*.csv
python -m src.diagnostics        # 2. multikollinearitás az eredeti adatokon
python -m src.run_experiments    # 3. 12 modell (+ q=2 változat) illesztése és értékelése
python -m src.plots              # 4. ábrák -> results/figures
```

A `run_experiments` a beágyazásokat a `results/embeddings` mappába menti, és újrafuttatáskor
újrahasznosítja őket (`--recompute` kapcsolóval újraszámol). A teljes futás néhány tíz perc,
a leglassabb lépés az ESS-en a 12 dimenziós (exact) t-SNE.

## Mappaszerkezet

```
data/raw/            eredeti Excel-állományok
data/processed/      tisztított CSV-k
src/config.py        beállítások (célváltozók, véletlenmag, hiperparaméterek)
src/data_cleaning.py adattisztítás (minden lépés naplózva: results/tables/cleaning_log.csv)
src/diagnostics.py   korreláció, VIF, kondíciószám, sajátértékek (1.4 fejezet)
src/reducers.py      PCA, MDS, t-SNE, autoencoder egységes felülettel
src/evaluation.py    értékelési mutatók
src/run_experiments.py  a kísérlet futtatása
src/plots.py         ábrák
results/tables/      eredménytáblák (CSV)
results/figures/     ábrák (PNG, 300 dpi)
results/embeddings/  az egyes módszerek által előállított alacsony dimenziós koordináták
```

## Módszertan röviden

### Adattisztítás

A tisztítás a hibás és hiányzó értékeket kezeli, a multikollinearitást **szándékosan megtartja**.
Kivétel a tökéletes kollinearitás (ESS: a megye a régióba ágyazott), mert ott az XᵀX nem invertálható.

- **Concrete:** 25 duplikált sor törlése; a kor (nap) ln-transzformációja a ferdeség miatt.
- **FAT:** a 42. megfigyelés hibás magassága (29,5 hüvelyk) a rögzített BMI-ből visszaszámolva 69,5-re javítva;
  a 0% testzsírú (élettanilag lehetetlen) sor törölve; átváltás kg-ra és cm-re.
- **ESS2020:** a célváltozó „Unknown” válaszainak törlése; `County` törlése; a munkaórák és az internethasználat
  hiánya „nem dolgozik” / „nem internetezik” jelentésű, ezért 0-val pótolva és indikátorral jelölve;
  irreális értékek (képzés > 25 év, munka > 100 óra/hét) kezelése; percben mért változók ln(1+x);
  kategorikus változók dummy-kódolása explicit referenciakategóriával.

### Dimenziószám

Fő beállítás: **Kaiser-kritérium** (korrelációs mátrix sajátértéke > 1), mind a négy módszerre ugyanaz a q
(Concrete: 4, FAT: 2, ESS: 12). Kiegészítő beállítás: **q = 2**, a vizualizációs eset, amire az MDS és a
t-SNE eredetileg készült.

### Módszerek

| Módszer | Megvalósítás | Fő beállítások |
|---|---|---|
| PCA | `sklearn.decomposition.PCA` (SVD) | standardizált adatok (korrelációs mátrix) |
| MDS | `sklearn.manifold.MDS`, metrikus, SMACOF | euklideszi távolság, 4 véletlen kezdőkonfiguráció |
| t-SNE | `sklearn.manifold.TSNE` | perplexitás 30, 1000 iteráció, PCA-inicializálás; q ≥ 4 esetén `exact` |
| Autoencoder | `sklearn.neural_network.MLPRegressor` (X → X) | p–16–q–16–p, tanh, Adam, MSE, 20% validáció, korai leállítás, 5 kezdősúly |

### Értékelési mutatók

| Csoport | Mutató | Mit mér |
|---|---|---|
| A) Multikollinearitás | max/átlag VIF, kondíciószám, max \|r\| | megszűnt-e a lineáris függőség az új változók között |
| B) Szerkezetmegőrzés | trustworthiness (k = 10) | lokális szomszédságok megőrzése |
| | Shepard-féle Spearman-korreláció | az eredeti és az új távolságok rangkorrelációja |
| | Stress-1 (optimális skálázással) | a távolságok eltérése (3.3 fejezet) |
| | visszaállítható variancia | az eredeti változók varianciájának lineárisan visszaállítható hányada (PCA-nál = kumulált magyarázott variancia) |
| C) Előrejelzés | CV R², RMSE, MAE / AUC, pontosság, log-loss | 5-szörös, 3-szor ismételt keresztvalidáció |
| D) Becslési stabilitás | szignifikáns együtthatók aránya | t-/z-próbák a teljes mintán (1.3 fejezet) |
| | bootstrap előjelváltási arány | milyen gyakran fordul meg egy együttható előjele (200 bootstrap-minta) |
| | együtthatók bootstrap-szórása | standardizált együtthatók instabilitása |
| E) Költség | illesztési idő (s) | |

Referenciaként minden adatbázison szerepel az **eredeti, multikollineáris változókra** illesztett OLS / logit modell.

**Transzduktív és induktív keresztvalidáció.** Az MDS és a t-SNE nem ad leképezést új megfigyelésekre
(3.4 és 4.4 fejezet), ezért az összehasonlítható protokollban a beágyazás a teljes X-en készül (a célváltozó
nélkül, így y-szivárgás nincs), és csak a ráépülő regressziót keresztvalidáljuk. A PCA-ra és az autoencoderre
emellett induktív CV is készül (`*_inductive` oszlopok), ahol a standardizálás és a dimenziócsökkentés is
csak a tanítóhajtáson tanul.

**Rangsor.** A `results/tables/ranks.csv` adatbázisonként és mutatócsoportonként átlagos rangot ad (1 = legjobb),
majd ezek átlagát (`rank_overall`).

## Eredmények

Lásd: `results/tables/results_all.csv`, `results/tables/ranks.csv` és `results/figures/`.

## Irodalom (a megvalósításhoz)

- Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
- van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *JMLR*, 9, 2579–2605.
- Venna, J., & Kaski, S. (2001). Neighborhood preservation in nonlinear projection methods. *ICANN 2001* (trustworthiness).
- Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. *Cement and Concrete Research*, 28(12).
- Johnson, R. W. (1996). Fitting percentage of body fat to simple body measurements. *Journal of Statistics Education*, 4(1).
