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

### 2.2 Analyse Exploratoire

```python
plt.figure(figsize=(15, 12))
sns.countplot(x='price_range', data=train_df)
plt.title('Price Range Distribution')
plt.show()

plt.figure(figsize=(20, 15))
categorical_features = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(2, 3, i)
    sns.countplot(data=train_df, x=feature, hue='price_range')
    plt.title(f'{feature} vs Price Range')
plt.tight_layout()
plt.show()

plt.figure(figsize=(20, 15))
sns.scatterplot(data=train_df, x='ram', y='price_range')
sns.scatterplot(data=train_df, x='battery_power', y='price_range')
plt.tight_layout()
plt.show()

numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(20, 20))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(6, 6, i)
    sns.kdeplot(train_df[feature], fill=True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
correlation = train_df.corr()['price_range'].sort_values(ascending=False)
sns.barplot(x=correlation.values, y=correlation.index)
plt.title("Top Features by Price Range")
plt.tight_layout()
plt.show()

top_5_features = correlation.index[1:6]
sns.pairplot(train_df[top_5_features.tolist() + ['price_range']], hue='price_range')
plt.show()
```

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

### 2.4 Préparation des Données pour le Machine Learning

```python
X = train_df_enhanced.drop('price_range', axis=1)
y = train_df_enhanced['price_range']

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

test_df_enhanced = test_df_enhanced[X.columns]
print(f"Final alignment confirmed: {all(X.columns == test_df_enhanced.columns)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
test_scaled = scaler.transform(test_df_enhanced)
```

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

### 2.6 Évaluation des Modèles

```python
plt.figure(figsize=(8, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Comparison")
plt.ylabel("Cross-Validation Accuracy")
plt.xlabel("Model")
plt.tight_layout()
plt.show()

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
pred = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10, 5))
residuals = y_test - pred
sns.histplot(residuals, kde=True)
plt.title("Distribution des résidus")
plt.xlabel("Résidus")
plt.show()
```