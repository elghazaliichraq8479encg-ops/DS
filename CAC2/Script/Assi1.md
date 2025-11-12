# COURS DE SCIENCE DES DONNÉES
## École Nationale de Commerce et de Gestion (ENCG) - 4ème Année
<img src="PHOTO1.jpeg" style="height:464px;margin-right:432px"/>

# EL GHAZALI ICHRAQ

## Nom de jeu de données Bank Marketing

# DESCREPTIF

 La base de données "Bank Marketing" provient d'une institution bancaire portugaise et est liée aux campagnes de marketing direct par téléphone menées par cette banque. L'objectif principal est de prédire si un client souscrira à un dépôt à terme (variable cible y) suite à ces appels.
 
 Cette base a été créée par Sérgio Moro, Pedro Cortez et Paulo Rita, et décrite dans un article intitulé "A Data-Driven Approach to Predict the Success of Bank Telemarketing" publié en 2014 dans la revue Decision Support Systems.
 
Le but est d'étudier et prédire l'efficacité des campagnes téléphoniques en analysant des données multiples sur les clients (âge, emploi, statut marital, etc.) et les détails des contacts marketing (durée de l'appel, nombre de contacts, résultats des campagnes précédentes, etc.). Ainsi, cette base de données sert à développer et tester des modèles de classification pour prédire la réussite des campagnes marketing par téléphone d'une banque en se basant sur des données réelles collectées entre 2008 et 2010.

**Code Python -INSTALLATION DU PACKAGE :**

```python

!pip install ucimlrepo
```
**Code Python -IMPORTATION DU DATA :**

```python

* `from ucimlrepo import fetch_ucirepo` → importe la fonction permettant de télécharger des jeux de données depuis la bibliothèque **UCI Machine Learning Repository**.
* `bank_marketing = fetch_ucirepo(id=222)` → télécharge le jeu de données **Bank Marketing** (ID 222 dans la base UCI).
* `X = bank_marketing.data.features` → extrait les **variables explicatives** (features) sous forme de DataFrame `X`.
* `y = bank_marketing.data.targets` → extrait la **variable cible** (target) sous forme de DataFrame `y`.
* `print(bank_marketing.metadata)` → affiche les **informations générales** sur le jeu de données (nom, description, source, etc.).
* `print(bank_marketing.variables)` → affiche la **description des variables** (noms, types, unités, etc.).


from ucimlrepo import fetch_ucirepo

# fetch dataset
bank_marketing = fetch_ucirepo(id=222)

# data (as pandas dataframes)
X = bank_marketing.data.features
y = bank_marketing.data.targets

# metadata
print(bank_marketing.metadata)

# variable information
print(bank_marketing.variables)
```
**Code Python -VISUALISATION AVEC GRAPHES :**

```python
categorical_features = ['job', 'marital', 'education']

plt.figure(figsize=(18, 6))
for i, feature in enumerate(categorical_features):
    plt.subplot(1, len(categorical_features), i + 1)
    sns.countplot(x=feature, data=df, palette='coolwarm')
    plt.title(f'Distribution of {feature.capitalize()}')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

```

* `categorical_features = ['job', 'marital', 'education']` → liste des variables catégorielles à analyser.
* `plt.figure(figsize=(18, 6))` → création de la figure avec une taille définie.
* `for i, feature in enumerate(categorical_features):` → boucle sur chaque variable.
* `plt.subplot(1, len(categorical_features), i + 1)` → création d’un sous-graphe pour chaque variable.
* `sns.countplot(x=feature, data=df, palette='coolwarm')` → graphique en barres des fréquences de chaque catégorie.
* `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.xticks()` → ajout des titres et labels, rotation des étiquettes.
* `plt.tight_layout()` → ajuste l’espace entre les graphiques.
* `plt.show()` → affiche les graphiques.



