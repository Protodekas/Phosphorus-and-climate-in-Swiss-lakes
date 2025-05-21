# Phosphorus and climate in swiss lakes
This repository contains an analysis of data on phosphorus in Swiss lakes, and on various climatic parameters. It is structured as follows:

- in the main folder can be found the excel files with the cleaned data used to carry out this analysis (.xlsx files), the python scripts used to carry out the analysis (.py files), the analysis report in jupyter notebook format (.ipynb file) and in text format (.pdf file)
- in the "raw data" folder can be found the original excel files as downloaded from opendata.swiss
- in the "Exploratory analysis outputs" and "Regression models outputs" folders can be found the files generated during the various stages of analysis

All these files and folders can be downloaded for reading on your personal computer, or opened directly in this github repository.


Structure:
# Analyse du lien entre le phosphore et le climat des lacs suisses

Ce projet vise à analyser l'impact de différentes variables climatiques (température, précipitations, ensoleillement, neige) sur les concentrations de phosphore dans plusieurs lacs suisses entre 1999 et 2020.

---

## 1. Importation des données

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importation des fichiers
phosphorus = pd.read_excel("phosphorus-lakes.xlsx", na_values=["..."])
weather = pd.read_excel("weather-data.xlsx", sheet_name=None, na_values=["..."])
# ...

## 2. Préparation et nettoyage des données

# Melting, mapping, merge, etc.
# Affichage d'extraits pour illustrer le traitement
phospho_weather.head()

## 3. Analyse exploratoire

### 3.1 Statistiques descriptives

phospho_weather.describe()

### 3.2 Corrélations entre phosphore et climat

for lake in phospho_weather["Lake"].unique():
    subset = phospho_weather[phospho_weather["Lake"] == lake]
    corr = subset[["Phosphorus", "Temperature", "Rainfall", "Insolation", "Snow"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title(f"Corrélation - {lake}")
    plt.show()

## 4. Modélisation

### 4.1 Régression linéaire

# Modèle linéaire pour chaque lac
# Affichage des R², coefficients, résidus, etc.

### 4.2 Régressions Ridge, Lasso, Random Forest

# Affichage des scores R² et RMSE
# Affichage des coefficients (via DataFrame)

## 5. Résultats et interprétations

- Les modèles linéaires s’ajustent assez bien aux données pour la majorité des lacs.
- Les modèles Ridge et Lasso permettent de limiter la surcouche de bruit et la multicolinéarité.
- Le modèle Random Forest n’apporte pas d'amélioration significative : complexité inutile ici.

### Exemples :
- Lake Zug : meilleur ajustement global (R² élevé)
- Lake Neuchâtel : modèle peu fiable (R² < 0)

## 6. Conclusion

- La température semble être la variable la plus liée aux niveaux de phosphore.
- Il reste néanmoins des facteurs non climatiques (usage agricole, gestion des eaux, etc.) à explorer.
- Ce projet montre l’intérêt d’une approche croisée climat + qualité de l’eau, mais aussi les limites d’interprétation d’un modèle basé uniquement sur des données agrégées.

---

Directories:
/data
/scripts
/notebooks
/figures
README.md
requirements.txt
