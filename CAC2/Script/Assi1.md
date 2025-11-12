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








