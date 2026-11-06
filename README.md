# Dimenziócsökkentő algoritmusok empirikus hatékonyságvizsgálata multikollinearitás kezelésében

Szakdolgozathoz tartozó kód: négy dimenziócsökkentő eljárás (PCA, MDS, t-SNE, autoencoder)
összehasonlítása három, multikollineáris magyarázóváltozókat tartalmazó adatbázison
(4 × 3 = 12 modell).

| Adatbázis | n | Célváltozó | Multikollinearitás eredete |
|---|---|---|---|
| Concrete (UCI) | 1030 | nyomószilárdság (MPa) | szerkezeti (keverékarányok) |
| FAT (testzsír) | 252 | testzsír-százalék | természetes (testméretek) |
| ESS2020 (HU) | 1849 | oltakozási hajlandóság | mérési (átfedő kérdések) |

## Telepítés

```bash
pip install -r requirements.txt
```
