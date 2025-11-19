##  Objectif du TP

L’objectif est d’apprendre un modèle de classification permettant de prédire la qualité du vin blanc à partir de caractéristiques physico-chimiques.
Le dataset provient de la base UCI et contient des mesures telles que l’acidité, le pH, le sucre résiduel, la densité…

Deux grandes étapes :

Analyse des données (Data Analysis)

Classification supervisée avec k-Nearest Neighbors (k-NN)
# Analyse des données
import pandas as pd
import numpy as np

link = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

df = pd.read_csv(link, header="infer", delimiter=";")
print("\n========= Dataset summary ========= \n")
df.info()
print("\n========= A few first samples ========= \n")
print(df.head())
import pandas as pd : importe la librairie pour manipuler des tableaux de données.

import numpy as np : importe NumPy pour des opérations numériques.

pd.read_csv(...) : lit le fichier CSV depuis l’URL.

delimiter=";" car les valeurs sont séparées par des points-virgules.

df.info() : affiche le nombre de lignes, de colonnes et les types de données.

df.head() : affiche les 5 premières lignes du dataset.
