# COURS DE SCIENCE DES DONNÉES
## École Nationale de Commerce et de Gestion (ENCG) - 4ème Année
# Analyse des facteurs influençant le prix des téléphones portables

## Sommaire

- [1. Contexte](#1-contexte)
- [2. Approche Méthodologique](#2-approche-méthodologique)
  - [2.1 Importation et Préparation des Données](#21-importation-et-préparation-des-données)
  - [2.2 Analyse Exploratoire](#22-analyse-exploratoire)
  - [2.3 Feature Engineering](#23-feature-engineering)
  - [2.4 Préparation des Données pour le Machine Learning](#24-préparation-des-données-pour-le-machine-learning)
  - [2.5 Modélisation](#25-modélisation)
  - [2.6 Évaluation des Modèles](#26-évaluation-des-modèles)

---

## 1. Contexte

Bob a fondé une entreprise de fabrication de téléphones portables. Pour se positionner face à Apple et Samsung, il souhaite déterminer la fourchette de prix de ses futurs modèles.

Il collecte pour cela un dataset contenant les caractéristiques techniques de plusieurs téléphones ainsi que leur catégorie de prix.

**Objectif :** Construire un modèle prédictif permettant d'associer chaque téléphone à une classe de prix :
- `0` = bas
- `1` = moyen
- `2` = haut
- `3` = premium

---

## 2. Approche Méthodologique

### 2.1 Importation et Préparation des Données

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Train dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)

print("\nTrain dataset info:")
print(train_df.info())
```

**Explication :**
Ce code importe les bibliothèques nécessaires pour l'analyse de données (pandas, numpy) et la visualisation (matplotlib, seaborn). Il configure le style visuel des graphiques et désactive les avertissements pour une sortie plus propre.

**Interprétation :**
Les fonctions `shape` et `info()` permettent de vérifier rapidement la structure du dataset (nombre de lignes/colonnes, types de données, valeurs manquantes). C'est une étape cruciale pour comprendre la qualité et la taille des données avant toute analyse.

---

### 2.2 Analyse Exploratoire

#### Distribution des catégories de prix

```python
plt.figure(figsize=(15, 12))
sns.countplot(x='price_range', data=train_df)
plt.title('Price Range Distribution')
plt.show()
```

**Explication :**
Crée un graphique en barres montrant combien de téléphones appartiennent à chaque catégorie de prix (0, 1, 2, 3).

**Interprétation :**
Ce graphique révèle si les données sont équilibrées entre les catégories. Un déséquilibre important pourrait nécessiter des techniques de rééquilibrage (oversampling/undersampling) pour éviter que le modèle ne favorise la classe majoritaire.

---

#### Analyse des caractéristiques catégorielles

```python
plt.figure(figsize=(20, 15))
categorical_features = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(2, 3, i)
    sns.countplot(data=train_df, x=feature, hue='price_range')
    plt.title(f'{feature} vs Price Range')
plt.tight_layout()
plt.show()
```

**Explication :**
Crée 6 sous-graphiques (2 lignes × 3 colonnes) montrant la relation entre chaque caractéristique binaire (Bluetooth, dual SIM, 4G, 3G, écran tactile, WiFi) et la catégorie de prix.

**Interprétation :**
Ces graphiques permettent d'identifier quelles fonctionnalités sont plus présentes dans les téléphones haut de gamme. Par exemple, si tous les téléphones premium ont la 4G, cette caractéristique est un bon prédicteur du prix.

---

#### Relation entre RAM/batterie et prix

```python
plt.figure(figsize=(20, 15))
sns.scatterplot(data=train_df, x='ram', y='price_range')
sns.scatterplot(data=train_df, x='battery_power', y='price_range')
plt.tight_layout()
plt.show()
```

**Explication :**
Crée deux nuages de points montrant la relation entre la RAM et le prix, puis entre la capacité de la batterie et le prix.

**Interprétation :**
Si les points forment une tendance ascendante claire, cela indique une forte corrélation positive : plus la RAM/batterie est élevée, plus le prix augmente. Ces variables pourraient être des prédicteurs importants.

---

#### Distribution des variables numériques

```python
numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(20, 20))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(6, 6, i)
    sns.kdeplot(train_df[feature], fill=True)
plt.tight_layout()
plt.show()
```

**Explication :**
Sélectionne toutes les colonnes numériques et crée un graphique de densité (KDE) pour chacune, permettant de visualiser leur distribution.

**Interprétation :**
Les graphiques KDE révèlent la forme de la distribution (normale, bimodale, asymétrique). Des distributions très asymétriques ou avec des valeurs aberrantes peuvent nécessiter une transformation (log, standardisation) avant la modélisation.

---

#### Corrélation avec le prix

```python
plt.figure(figsize=(10, 8))
correlation = train_df.corr()['price_range'].sort_values(ascending=False)
sns.barplot(x=correlation.values, y=correlation.index)
plt.title("Top Features by Price Range")
plt.tight_layout()
plt.show()
```

**Explication :**
Calcule la corrélation de Pearson entre toutes les variables numériques et la catégorie de prix, puis affiche les résultats triés dans un graphique en barres.

**Interprétation :**
Les variables avec une corrélation élevée (proche de 1 ou -1) ont un impact fort sur le prix. Ce graphique aide à identifier les features les plus importantes pour la prédiction et à éliminer les variables non pertinentes.

---

#### Analyse des relations entre top features

```python
top_5_features = correlation.index[1:6]
sns.pairplot(train_df[top_5_features.tolist() + ['price_range']], hue='price_range')
plt.show()
```

**Explication :**
Sélectionne les 5 variables les plus corrélées avec le prix et crée une matrice de nuages de points montrant toutes les relations possibles entre ces variables, colorées par catégorie de prix.

**Interprétation :**
Le pairplot révèle les interactions entre variables : multicolinéarité (deux variables très corrélées entre elles), séparabilité des classes (les couleurs sont-elles bien séparées ?), et patterns non-linéaires qui pourraient nécessiter des transformations.

---

### 2.3 Feature Engineering

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

def create_features(df):
    df = df.copy()
    df['pixel_resolution'] = df['px_height'] * df['px_width']
    df['screen_size'] = df['sc_h'] * df['sc_w']
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

print("Applying feature engineering...")
train_df_enhanced = create_features(train_df)
test_df_enhanced = create_features(test_df)

print(f"Enhanced train shape: {train_df_enhanced.shape}")
print(f"Enhanced test shape: {test_df_enhanced.shape}")
```

**Explication :**
Importe les outils de machine learning nécessaires et définit une fonction qui crée deux nouvelles variables : la résolution totale de l'écran (hauteur × largeur en pixels) et la taille de l'écran (hauteur × largeur en cm). Elle remplace également les valeurs infinies par NaN.

**Interprétation :**
Le feature engineering consiste à créer de nouvelles variables plus significatives que les variables brutes. Par exemple, la résolution totale (12 mégapixels) est plus pertinente que la hauteur et la largeur séparément. Ces features combinées peuvent améliorer significativement la performance du modèle.

---

#### Vérification de la cohérence des datasets

```python
test_df_enhanced['pixel_resolution'] = test_df_enhanced['px_width'] * test_df_enhanced['px_height']
test_df_enhanced['screen_size'] = test_df_enhanced['sc_h'] * test_df_enhanced['sc_w']

train_features = [col for col in train_df_enhanced.columns if col != 'price_range']
test_features = test_df_enhanced.columns.tolist()

print(f"Features match: {set(train_features) == set(test_features)}")

missing_in_test = set(train_features) - set(test_features)
missing_in_train = set(test_features) - set(train_features)

print(f"Features missing in test: {missing_in_test}")
print(f"Features missing in train: {missing_in_train}")
```

**Explication :**
Applique les mêmes transformations au dataset de test, puis vérifie que les colonnes des datasets d'entraînement et de test sont identiques (sauf la colonne cible 'price_range').

**Interprétation :**
C'est une étape critique : si le test set a des colonnes différentes, le modèle entraîné ne pourra pas faire de prédictions. Cette vérification prévient les erreurs lors du déploiement et garantit que les données de production auront la même structure que les données d'entraînement.

---

### 2.4 Préparation des Données pour le Machine Learning

```python
X = train_df_enhanced.drop('price_range', axis=1)
y = train_df_enhanced['price_range']

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

test_df_enhanced = test_df_enhanced[X.columns]
print(f"Final alignment confirmed: {all(X.columns == test_df_enhanced.columns)}")
```

**Explication :**
Sépare le dataset en features (X = toutes les colonnes sauf le prix) et target (y = catégorie de prix). Réorganise ensuite le test set pour avoir exactement les mêmes colonnes dans le même ordre que X.

**Interprétation :**
Cette séparation est fondamentale en machine learning : X contient les informations que le modèle utilisera pour apprendre, y contient ce qu'il doit prédire. L'alignement des colonnes garantit que le modèle reçoit les données dans le format attendu.

---

#### Split et normalisation

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
test_scaled = scaler.transform(test_df_enhanced)
```

**Explication :**
Divise les données en 80% entraînement et 20% test, en respectant la proportion des classes (stratify). Puis standardise toutes les variables numériques (moyenne=0, écart-type=1).

**Interprétation :**
Le split permet d'évaluer le modèle sur des données qu'il n'a jamais vues. Le paramètre `stratify` est crucial pour les problèmes de classification : il garantit que chaque catégorie de prix est représentée proportionnellement dans train et test. La standardisation est nécessaire car certains algorithmes (SVM, régression logistique) sont sensibles à l'échelle des variables : sans elle, une variable avec de grandes valeurs (ex: RAM en Mo) dominerait une variable avec de petites valeurs (ex: nombre de cœurs).

---

### 2.5 Modélisation

```python
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    results[name] = cv_scores.mean()
    print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
```

**Explication :**
Définit deux modèles de classification (Random Forest et Naive Bayes) et les évalue avec une validation croisée à 5 plis (5-fold cross-validation). Chaque modèle est entraîné 5 fois sur différentes partitions des données.

**Interprétation :**
La validation croisée donne une estimation plus robuste de la performance que le simple train/test split : elle utilise toutes les données pour l'entraînement ET le test, réduisant le risque de sur-apprentissage. L'écart-type indique la stabilité du modèle : un écart-type élevé signifie que la performance varie beaucoup selon les données, ce qui peut être problématique en production. Random Forest est généralement plus performant pour ce type de problème car il capture les interactions non-linéaires entre variables.

---

### 2.6 Évaluation des Modèles

#### Comparaison visuelle des modèles

```python
plt.figure(figsize=(8, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Comparison")
plt.ylabel("Cross-Validation Accuracy")
plt.xlabel("Model")
plt.tight_layout()
plt.show()
```

**Explication :**
Crée un graphique en barres comparant la précision moyenne de chaque modèle obtenue par validation croisée.

**Interprétation :**
Ce graphique permet de sélectionner visuellement le meilleur modèle. Une différence de plus de 5% entre deux modèles est généralement significative. Le choix final dépend aussi d'autres critères : temps d'entraînement, interprétabilité, facilité de déploiement.

---

#### Matrice de confusion

```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

**Explication :**
Entraîne un Random Forest sur l'ensemble d'entraînement, fait des prédictions sur le test set, puis affiche une matrice de confusion visualisant les erreurs de classification.

**Interprétation :**
La matrice de confusion révèle où le modèle se trompe : la diagonale montre les prédictions correctes, les autres cellules montrent les confusions entre classes. Par exemple, si le modèle confond souvent les prix "moyen" et "haut", cela indique que ces catégories ont des caractéristiques similaires. Cette analyse guide les améliorations : ajouter des features différenciatrices, collecter plus de données pour les classes problématiques, ou ajuster les seuils de décision.

---

#### Distribution des résidus

```python
plt.figure(figsize=(10, 5))
residuals = y_test - pred
sns.histplot(residuals, kde=True)
plt.title("Distribution des résidus")
plt.xlabel("Résidus")
plt.show()
```

**Explication :**
Calcule les résidus (différence entre valeur réelle et prédiction) et affiche leur distribution sous forme d'histogramme avec une courbe de densité.

**Interprétation :**
Pour un problème de classification, les résidus devraient être majoritairement à 0 (prédictions correctes). Un pic à 0 indique un bon modèle. Des pics à -1, 1, -2, 2, etc. montrent les erreurs : par exemple, un résidu de 1 signifie que le modèle a prédit une catégorie trop basse. Si les résidus suivent un pattern spécifique (ex: toujours positifs pour une certaine gamme), cela suggère un biais systématique du modèle qui pourrait être corrigé.
