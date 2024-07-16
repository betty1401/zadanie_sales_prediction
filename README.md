# Big Mart Sales Prediction

Tento repozitár obsahuje moje riešnie k zadaniu Big Mart Sales Prediction, ktorý využíva [tento dataset](https://www.kaggle.com/datasets/shivan118/big-mart-sales-prediction-datasets). Projekt sa skladá z dvoch častí, a to:

1. Exploratory Data Analysis (eda.py)
2. Prediction Model (model.py)

## Exploratory Data Analysis
Prvá časť sa venuje preskúmaniu datasetu a jeho preprocessingu. Ako prvé som zisťovala základné vlastnosti dát, a to z akých stĺpov sa skladá, dátové typy dát v stĺpcoch, koľko a aké unikátne hodnoty sú v samotných stĺpcoch, výskyt nulových hodnôt a základné informácie cez deskriptívnu štatistiku. Na základe dátového typu som identifikovala kategorické dáta, ktoré som vizualizovala cez bar chart. Na základe týchto krokov som identifikovala chyby, ktoré treba upraviť pred trénovaním samotného modelu. Kroky prípravy a preprocessingu dát boli: 
1. zjednotenie názvov pre kategórie,
2. nahradenie nulových hodnôt - priemerom v prípade numerických dát a najfrekventovanejším stringom v prípade kategorických dát,
3. transformácia kategorických dát na numerické hodnoty cez LabelEncoding,
4. vyhľadanie korelácie medzi premennými pre identifikáciu správnych hodnôt k trénovanie modelu,
5. identifikácia a odstránenie outliers.

Po týchto úpravách som dataset nanovo uložila kvôli zachovaniu aj pôvodného datasetu.

## Trénovanie modelu
Na vytvorenie predikčného modelu som vybrala regresný model založený na algoritme Random Forest. Ako prvé som opäť pripravila dáta - určila si trénovacie premenné a výslednú premennú. Dataset som rozložila štandardne na 80/20. Ako prvé som trénovala model s pôvodnými parametrami, ale tam mi výsledok vyšiel trénovacie skóre 98% a testovacie skóre 49%. Takýto veľký rozdiel naznačuje overfitting, preto som zvolila metódu hypertuningu samotných parametrov modela. K tomuto som zvolila algoritmus RandomizedSearch a podľa neho som nastavila s menšími úpravami parametre modelu. 

Výsledky modelu boli:
- *MAE:* 756.47
- *R^2:* 0.58
- *Trénovacie skóre:* 0.82
- *Testovacie skóre:* 0.58
- *RMSE skóre cross-validácie pre 3 behy:* [1143.02, 1119.11, 1130.61]

Z výsledkov je vidieť, že odchýlka predikovanej hodnoty k reálnej je 756.47 jednotiek meny. Podarilo sa mi znížiť overfitting efekt, ale stále to môže v nejakej miere pretrvávať (to je nevýhoda aj tohto typu modelu). Tiež je vidieť, že počas troch behov odchýlka zostáva pomerne konzistentná a nie sú v nich veľké výkyvy. Celkovo model má pomerne dobrý výsledok na základe veľkosti dát a použitých metód. 
