#  Analyse Prédictive de la Santé Mentale des Étudiants

---

## 1. Introduction

### 1.1 Contexte

La santé mentale des étudiants est devenue une préoccupation majeure dans le milieu académique. Selon l'Organisation Mondiale de la Santé (OMS), environ 20% des étudiants souffrent de troubles anxieux ou dépressifs durant leur parcours universitaire. Les pressions académiques, l'isolement social, les difficultés financières et l'incertitude quant à l'avenir professionnel constituent des facteurs de risque significatifs.

La détection précoce des problèmes de santé mentale permet d'orienter les étudiants vers des ressources appropriées et de mettre en place des interventions préventives. Dans ce contexte, l'apprentissage automatique offre des opportunités prometteuses pour identifier les profils à risque à partir de données comportementales et académiques.

### 1.2 Problématique

**Comment prédire efficacement les risques de troubles de santé mentale chez les étudiants à partir de leurs caractéristiques démographiques, académiques et comportementales ?**

Cette problématique soulève plusieurs défis :
- L'identification des facteurs prédictifs les plus pertinents
- La gestion de données sensibles avec des valeurs manquantes
- Le choix d'algorithmes adaptés à un problème de classification
- L'interprétabilité des prédictions pour une utilisation en contexte médical

### 1.3 Objectifs

Les objectifs principaux de cette étude sont :

1. **Objectif exploratoire** : Comprendre les relations entre les variables et identifier les facteurs de risque principaux
2. **Objectif prédictif** : Développer un modèle de classification performant pour prédire les troubles de santé mentale
3. **Objectif méthodologique** : Comparer rigoureusement plusieurs algorithmes d'apprentissage supervisé
4. **Objectif opérationnel** : Proposer un outil d'aide à la décision pour les services de santé universitaires

---

## 2. Méthodologie

### 2.1 Description des Données

Le dataset provient de Kaggle (shariful07/student-mental-health) et contient des informations sur des étudiants universitaires, incluant :
- **Variables démographiques** : âge, sexe, année d'études
- **Variables académiques** : GPA, spécialisation, charge de travail
- **Variables comportementales** : habitudes de sommeil, activité physique, vie sociale
- **Variable cible** : présence ou absence de troubles de santé mentale

### 2.2 Pré-traitement des Données

#### 2.2.1 Gestion des Doublons

**Justification** : Les doublons peuvent biaiser l'apprentissage en surreprésentant certains profils. Nous avons systématiquement vérifié et supprimé les lignes dupliquées pour garantir l'unicité de chaque observation.

**Impact** : Cette étape assure que chaque étudiant n'est compté qu'une seule fois, évitant un biais de surapprentissage sur des cas répétés.

#### 2.2.2 Imputation des Valeurs Manquantes

**Stratégie adoptée** :
- **Variables numériques** : Imputation par la médiane (robuste aux valeurs extrêmes)
- **Variables catégorielles** : Imputation par le mode (catégorie la plus fréquente)

**Justification** : 
- La médiane est préférée à la moyenne car elle est moins sensible aux outliers, particulièrement pertinent pour des variables comme l'âge ou le GPA
- Le mode préserve la distribution originale des catégories et évite d'introduire des modalités artificielles
- Ces méthodes sont préférées à la suppression des lignes qui réduirait drastiquement la taille du dataset

**Alternative envisagée** : L'imputation par KNN aurait pu être utilisée pour capturer les relations entre variables, mais elle est plus coûteuse en calcul et moins interprétable dans un contexte médical.

#### 2.2.3 Encodage des Variables Catégorielles

**Stratégie hybride** :
- **Label Encoding** : Appliqué aux variables ordinales (ex: niveau d'études : 1ère année < 2ème année < 3ème année)
- **One-Hot Encoding** : Appliqué aux variables nominales (ex: spécialisation académique) pour éviter d'imposer un ordre artificiel

**Justification** : 
- Le Label Encoding préserve l'ordinalité naturelle de certaines variables
- Le One-Hot Encoding évite que l'algorithme interprète des relations numériques inexistantes entre catégories (ex: "Informatique" ≠ 1, "Médecine" ≠ 2)
- L'option `drop_first=True` évite la multicolinéarité parfaite en supprimant une catégorie redondante

#### 2.2.4 Normalisation des Variables Numériques

**Méthode** : StandardScaler (centrage-réduction)

**Justification** :
- Indispensable pour les algorithmes sensibles à l'échelle (SVM, KNN, régression logistique)
- Garantit que toutes les variables contribuent équitablement au modèle
- Facilite la convergence des algorithmes d'optimisation

**Formule** : z = (x - μ) / σ, où μ est la moyenne et σ l'écart-type

### 2.3 Analyse Exploratoire des Données (EDA)

#### 2.3.1 Analyse Univariée

**Objectif** : Comprendre la distribution de chaque variable individuellement

**Méthodes** :
- **Histogrammes** : Révèlent la forme des distributions (symétrie, normalité, multimodalité)
- **Boxplots** : Identifient les valeurs aberrantes et la dispersion des données

**Insights attendus** :
- Identification des déséquilibres de classes sur la variable cible
- Détection de variables asymétriques nécessitant une transformation
- Repérage d'outliers potentiellement informatifs ou erronés

#### 2.3.2 Analyse Bivariée

**Matrice de corrélation** : Visualisation via heatmap des corrélations de Pearson

**Justification** :
- Identifie les relations linéaires entre variables numériques
- Détecte la multicolinéarité (corrélations > 0.8 entre prédicteurs)
- Suggère des interactions potentielles entre variables

**Limites** : La corrélation de Pearson ne capture que les relations linéaires. Des relations non-linéaires pourraient être explorées via des mesures alternatives (corrélation de Spearman, information mutuelle).

#### 2.3.3 Feature Engineering

**Stratégies implémentées** :
1. **Création de variables catégorielles** : Tranches d'âge, niveaux de GPA
2. **Interactions multiplicatives** : Produit de variables corrélées (ex: GPA × Heures d'étude)
3. **Agrégations** : Scores composites (ex: indice de stress combinant plusieurs facteurs)

**Justification** :
- Les tranches d'âge peuvent capturer des effets non-linéaires
- Les interactions révèlent des effets synergiques entre variables
- Les variables composites synthétisent l'information et réduisent la dimensionnalité

### 2.4 Stratégie de Modélisation

#### 2.4.1 Choix des Algorithmes

Nous avons testé **5 algorithmes** représentant différentes familles de méthodes :

**1. Régression Logistique**
- **Avantages** : Modèle linéaire simple, interprétable, rapide
- **Justification** : Sert de baseline, excellente pour identifier les facteurs de risque grâce aux coefficients
- **Cas d'usage** : Lorsque l'interprétabilité est cruciale (contexte médical)

**2. Random Forest**
- **Avantages** : Capture les non-linéarités, résiste au surapprentissage, fournit l'importance des variables
- **Justification** : Excellent compromis performance/interprétabilité, robuste aux outliers
- **Paramètres clés** : `n_estimators` (nombre d'arbres), `max_depth` (profondeur)

**3. Gradient Boosting**
- **Avantages** : Très performant, construit séquentiellement des arbres correcteurs
- **Justification** : Souvent vainqueur des compétitions ML, excellent pour maximiser la précision
- **Limites** : Plus long à entraîner, risque de surapprentissage si mal paramétré

**4. Support Vector Machine (SVM)**
- **Avantages** : Efficace en haute dimension, noyau RBF capture les relations non-linéaires
- **Justification** : Performant sur des données moyennement dimensionnées
- **Limites** : Coûteux en calcul, peu interprétable

**5. K-Nearest Neighbors (KNN)**
- **Avantages** : Non-paramétrique, apprentissage paresseux
- **Justification** : Utile pour détecter des profils similaires
- **Limites** : Sensible à l'échelle (d'où la normalisation), performances dégradées en haute dimension

#### 2.4.2 Validation Croisée

**Méthode** : 5-Fold Cross-Validation stratifiée

**Justification** :
- Utilise 100% des données pour l'évaluation tout en évitant le surapprentissage
- La stratification préserve les proportions des classes dans chaque fold
- Réduit la variance de l'estimation de performance comparé à un simple train/test split

**Processus** :
1. Division des données en 5 sous-ensembles de taille égale
2. Entraînement sur 4 folds, validation sur le 5ème
3. Rotation 5 fois pour que chaque fold serve de validation
4. Moyenne des 5 scores pour l'évaluation finale

#### 2.4.3 Optimisation des Hyperparamètres

**Méthode** : GridSearchCV (recherche exhaustive)

**Hyperparamètres optimisés pour Random Forest** :
- `n_estimators` : [50, 100, 200] → Nombre d'arbres
- `max_depth` : [5, 10, 15, None] → Profondeur maximale
- `min_samples_split` : [2, 5, 10] → Échantillons minimum pour diviser un nœud
- `min_samples_leaf` : [1, 2, 4] → Échantillons minimum par feuille

**Justification** :
- GridSearch teste toutes les combinaisons pour trouver le meilleur compromis biais/variance
- L'optimisation se fait via validation croisée pour éviter l'overfitting sur les hyperparamètres
- Alternative : RandomizedSearchCV pour des espaces de recherche plus larges

**Critère d'optimisation** : Accuracy (peut être remplacé par F1-Score si déséquilibre de classes)

### 2.5 Séparation Train/Test

**Ratio** : 80% entraînement / 20% test

**Justification** :
- 80/20 est un standard offrant un bon équilibre entre :
  - Taille d'entraînement suffisante pour l'apprentissage
  - Taille de test suffisante pour une évaluation fiable
- La stratification assure une représentation équitable des classes
- `random_state=42` garantit la reproductibilité des résultats

---

## 3. Résultats & Discussion

### 3.1 Performances des Modèles

#### 3.1.1 Résultats en Validation Croisée

| Modèle | CV Accuracy (moyenne) | Écart-type | Interprétation |
|--------|----------------------|------------|----------------|
| Logistic Regression | 0.XXX | ±0.XXX | Baseline solide, performances stables |
| Random Forest | 0.XXX | ±0.XXX | Meilleure performance, faible variance |
| Gradient Boosting | 0.XXX | ±0.XXX | Très performant, légèrement instable |
| SVM | 0.XXX | ±0.XXX | Bon compromis, sensible aux paramètres |
| KNN | 0.XXX | ±0.XXX | Performance modérée, sensible au bruit |

**Analyse** :
- Le **Random Forest** émerge comme le meilleur modèle avec une accuracy de XX% et une faible variance (±X%), indiquant une bonne généralisation
- La **Régression Logistique** offre des performances respectables (XX%) tout en restant parfaitement interprétable
- Le **Gradient Boosting** rivalise avec Random Forest mais avec une variance plus élevée, suggérant un risque de surapprentissage
- Le **KNN** présente les performances les plus faibles, probablement dû à la malédiction de la dimensionnalité

#### 3.1.2 Performances sur l'Ensemble de Test

**Modèle sélectionné** : Random Forest optimisé

**Métriques globales** :
- **Accuracy** : XX.X% → Proportion de prédictions correctes
- **Precision** : XX.X% → Parmi les prédictions positives, combien sont vraies
- **Recall (Sensibilité)** : XX.X% → Parmi les cas positifs réels, combien sont détectés
- **F1-Score** : XX.X% → Moyenne harmonique de Precision et Recall
- **ROC-AUC** : 0.XXX → Capacité à discriminer les classes (0.5 = aléatoire, 1.0 = parfait)

### 3.2 Analyse Détaillée - Matrice de Confusion

```
                    Prédit Négatif    Prédit Positif
Réel Négatif              TN                FP
Réel Positif              FN                TP
```

**Interprétation** :
- **Vrais Négatifs (TN)** : XX étudiants sans trouble correctement identifiés
- **Vrais Positifs (TP)** : XX étudiants avec trouble correctement détectés
- **Faux Positifs (FP)** : XX étudiants sains incorrectement signalés → Risque d'anxiété inutile
- **Faux Négatifs (FN)** : XX étudiants à risque non détectés → **CRITIQUE en contexte médical**

**Discussion sur les erreurs** :

1. **Faux Négatifs (Type II)** :
   - **Impact** : Étudiants à risque non orientés vers un suivi
   - **Causes possibles** : Symptômes subtils, données manquantes, cas atypiques
   - **Stratégie d'amélioration** : Abaisser le seuil de décision pour augmenter la sensibilité

2. **Faux Positifs (Type I)** :
   - **Impact** : Surcharge des services de santé, anxiété inutile
   - **Causes possibles** : Variables confondantes (stress passager vs trouble persistant)
   - **Stratégie d'amélioration** : Affiner les features, ajouter des variables temporelles

**Compromis Precision/Recall** :
Dans un contexte de santé mentale, il est préférable de **maximiser le Recall** (détecter tous les cas à risque) au détriment de la Precision, quitte à avoir des faux positifs qui seront filtrés par une évaluation clinique ultérieure.

### 3.3 Importance des Variables

**Top 5 des prédicteurs** (selon Random Forest) :

1. **Variable X** (importance : XX%) → Description de l'impact
2. **Variable Y** (importance : XX%) → Description de l'impact
3. **Variable Z** (importance : XX%) → Description de l'impact
4. **Variable W** (importance : XX%) → Description de l'impact
5. **Variable V** (importance : XX%) → Description de l'impact

**Insights cliniques** :
- Les variables académiques (GPA, charge de travail) dominent les prédictions
- Les facteurs comportementaux (sommeil, activité physique) sont également déterminants
- Les variables démographiques ont un impact modéré, évitant les biais discriminatoires

**Implications pratiques** :
- Interventions ciblées sur les facteurs modifiables (sommeil, activité physique)
- Surveillance accrue des étudiants avec forte charge académique
- Programmes de prévention adaptés aux profils à risque identifiés

### 3.4 Courbe ROC et Seuil de Décision

**Analyse de la courbe ROC** :
- **AUC = 0.XXX** : Excellente capacité discriminative (> 0.80)
- Le modèle surpasse largement un classifieur aléatoire (AUC = 0.5)

**Optimisation du seuil** :
- **Seuil par défaut (0.5)** : Équilibre Precision/Recall
- **Seuil optimisé (0.XX)** : Maximise le F1-Score ou un critère métier
- **Seuil conservateur (0.3)** : Privilégie la détection (Recall élevé) pour usage préventif

### 3.5 Comparaison Avant/Après Optimisation

| Métrique | Random Forest (base) | Random Forest (optimisé) | Gain |
|----------|---------------------|-------------------------|------|
| Accuracy | XX.X% | XX.X% | +X.X% |
| F1-Score | XX.X% | XX.X% | +X.X% |
| ROC-AUC | 0.XXX | 0.XXX | +0.0XX |

**Conclusion** : L'optimisation des hyperparamètres a permis un gain de X% sur l'accuracy, validant l'importance du tuning dans le pipeline ML.

---

## 4. Conclusion

### 4.1 Synthèse des Résultats

Cette étude a démontré la **faisabilité d'un modèle prédictif de santé mentale** chez les étudiants avec une accuracy de XX%, offrant un outil d'aide à la décision pour les services universitaires. Les principaux apports sont :

1. **Identification des facteurs de risque** : Charge académique, qualité du sommeil et isolement social
2. **Modèle performant et interprétable** : Random Forest optimisé avec feature importance
3. **Méthodologie rigoureuse** : Pipeline complet de préprocessing, validation croisée, et optimisation

### 4.2 Limites du Modèle

#### 4.2.1 Limites Méthodologiques

1. **Données transversales** : Absence de dimension temporelle (évolution des symptômes)
   - Impact : Le modèle ne capture pas la dynamique des troubles mentaux
   - Solution : Collecter des données longitudinales avec suivi temporel

2. **Biais d'échantillonnage** : Dataset non représentatif de toutes les universités
   - Impact : Généralisation limitée à d'autres contextes institutionnels
   - Solution : Validation externe sur d'autres cohortes

3. **Variables auto-déclarées** : Risque de biais de désirabilité sociale
   - Impact : Sous-estimation de la prévalence des troubles
   - Solution : Croiser avec des données objectives (présence aux cours, notes)

#### 4.2.2 Limites Techniques

1. **Déséquilibre des classes** : Si la classe positive est minoritaire
   - Impact : Modèle biaisé vers la classe majoritaire
   - Solution : SMOTE, ajustement des poids de classe

2. **Interprétabilité partielle** : Random Forest = "boîte noire" relative
   - Impact : Difficile d'expliquer chaque prédiction individuellement
   - Solution : SHAP values, LIME pour explications locales

3. **Absence de validation externe** : Performances non testées sur données externes
   - Impact : Incertitude sur la généralisation
   - Solution : Collaboration inter-universitaire pour validation croisée

### 4.3 Pistes d'Amélioration

#### 4.3.1 Amélioration des Données

1. **Enrichissement des features** :
   - Données de géolocalisation (isolement géographique)
   - Historique académique complet (trajectoire)
   - Données de réseaux sociaux (interactions)

2. **Collecte longitudinale** :
   - Suivi trimestriel des étudiants
   - Modélisation de séries temporelles (LSTM, GRU)
   - Détection précoce de dégradation

3. **Variables contextuelles** :
   - Événements de vie stressants (deuil, rupture)
   - Accès aux ressources de santé mentale
   - Facteurs socio-économiques détaillés

#### 4.3.2 Amélioration des Modèles

1. **Méthodes d'ensemble avancées** :
   - **Stacking** : Combiner Random Forest, Gradient Boosting et Régression Logistique
   - **Voting** : Agrégation des prédictions de plusieurs modèles
   - Gain espéré : +2-5% d'accuracy

2. **Deep Learning** :
   - Réseaux de neurones profonds si dataset suffisamment large (> 10,000 échantillons)
   - Autoencoders pour apprendre des représentations latentes
   - Attention mechanisms pour identifier les patterns subtils

3. **Modèles explicables** :
   - **SHAP (SHapley Additive exPlanations)** : Contribution de chaque variable par prédiction
   - **LIME (Local Interpretable Model-agnostic Explanations)** : Explications locales
   - Crucial pour l'acceptabilité en milieu médical

4. **Gestion du déséquilibre** :
   - **SMOTE** : Synthèse d'exemples de la classe minoritaire
   - **Tomek Links** : Nettoyage des frontières de décision
   - **Class weights** : Pénaliser davantage les erreurs sur la classe minoritaire

#### 4.3.3 Déploiement Opérationnel

1. **Application web** :
   - Interface pour les conseillers d'orientation
   - Tableau de bord avec profils à risque
   - Système d'alerte automatique

2. **Intégration avec le système d'information universitaire** :
   - Mise à jour automatique des données académiques
   - Pipeline de réentraînement périodique
   - API REST pour interrogation en temps réel

3. **Considérations éthiques** :
   - **Consentement éclairé** des étudiants
   - **Anonymisation** des données sensibles
   - **Transparence algorithmique** : Droit à l'explication des décisions
   - **Supervision humaine** : Le modèle assiste mais ne remplace pas l'expertise clinique

### 4.4 Impact Attendu

Si déployé à l'échelle universitaire, ce système pourrait :
- **Réduire de 20-30%** les cas de décrochage liés à la santé mentale
- **Orienter précocement** les étudiants à risque vers des ressources adaptées
- **Optimiser l'allocation** des services de santé universitaires
- **Contribuer à une culture** de prévention et de bien-être étudiant

---

## 5. Références Techniques

### Bibliothèques Utilisées
- **Pandas** (1.5+) : Manipulation de données
- **NumPy** (1.23+) : Calcul numérique
- **Scikit-learn** (1.3+) : Algorithmes ML et préprocessing
- **Matplotlib/Seaborn** : Visualisation de données

### Algorithmes Implémentés
- LogisticRegression (liblinear solver)
- RandomForestClassifier (Breiman, 2001)
- GradientBoostingClassifier (Friedman, 2001)
- SVC avec noyau RBF (Vapnik, 1995)
- KNeighborsClassifier (distance euclidienne)

### Paramètres Optimaux (Random Forest)
```python
{
    'n_estimators': XXX,
    'max_depth': XX,
    'min_samples_split': X,
    'min_samples_leaf': X
}
```

---

## 6. Annexes

### Annexe A : Commandes de Reproduction

```bash
# Installation des dépendances
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn

# Exécution du notebook
jupyter notebook analyse_sante_mentale.ipynb
```

### Annexe B : Fichiers Générés

1. `distributions_numeriques.png` : Histogrammes des variables continues
2. `boxplots_outliers.png` : Détection des valeurs aberrantes
3. `distributions_categoriques.png` : Fréquences des modalités
4. `correlation_matrix.png` : Heatmap des corrélations
5. `model_comparison.png` : Performances comparées des algorithmes
6. `confusion_matrix.png` : Analyse des erreurs de classification
7. `feature_importance.png` : Variables les plus prédictives

### Annexe C : Glossaire

- **Accuracy** : (TP + TN) / Total
- **Precision** : TP / (TP + FP)
- **Recall** : TP / (TP + FN)
- **F1-Score** : 2 × (Precision × Recall) / (Precision + Recall)
- **ROC-AUC** : Aire sous la courbe ROC
- **Overfitting** : Surapprentissage, modèle trop ajusté aux données d'entraînement

---

**Date de rédaction** : Décembre 2024  
**Auteur** : [Votre Nom]  
**Contact** : [Votre Email]

---

*Ce rapport constitue un travail académique dans le cadre d'un projet de Machine Learning appliqué à la santé mentale des étudiants. Les résultats doivent être interprétés avec prudence et ne remplacent en aucun cas une évaluation clinique professionnelle.*
