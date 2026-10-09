
# 🧩 Segmentation des clients — RFM, K-Means et Random Forest

Projet de Machine Learning de bout en bout sur le dataset **Online Retail** : à partir de transactions e-commerce, on construit une vue **par client** (RFM), on identifie **3 segments** avec **K-Means**, puis on entraîne un **Random Forest** capable d'affecter automatiquement un nouveau client à l'un de ces segments. Le modèle est ensuite exposé via une application **Streamlit**.

```
Données transactionnelles → Nettoyage → RFM → Log + Standardisation
→ Recherche de K (Elbow / Silhouette) → Contrainte métier K = 3 → K-Means
→ Profiling & interprétation métier → PCA (visualisation)
→ Random Forest → Évaluation → Prédiction d'un nouveau client
→ Sauvegarde du modèle → Application Streamlit
```

---

## 📑 Sommaire

1. [Objectifs](#-objectifs)
2. [Données](#-données)
3. [Structure du projet](#-structure-du-projet)
4. [Installation](#-installation)
5. [Méthodologie](#-méthodologie)
6. [Résultats](#-résultats)
7. [Application Streamlit](#-application-streamlit)
8. [Utiliser le modèle sauvegardé](#-utiliser-le-modèle-sauvegardé)
9. [Limites et pistes d&#39;amélioration](#-limites-et-pistes-damélioration)

---

## 🎯 Objectifs

- Transformer des lignes de transactions en une **table client** (`CustomerID | Recence | Frequence | Montant`).
- Identifier des profils de clients par **clustering non supervisé** (K-Means).
- Respecter la **contrainte métier : exactement 3 segments** (K = 3).
- Entraîner un **classifieur supervisé** (Random Forest) qui apprend à reproduire cette segmentation.
- Permettre à un utilisateur non technique de **prédire le segment d'un nouveau client** via Streamlit.

---

## 📦 Données

| Élément    | Valeur                                                           |
| ------------ | ---------------------------------------------------------------- |
| Source       | Dataset**Online Retail** (transactions d'un e-commerçant) |
| Fichier      | `data/data.xlsx`                                               |
| Période     | 1er décembre 2010 → 9 décembre 2011                           |
| Taille brute | 541 909 lignes × 8 colonnes                                     |

**Colonnes :** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.

### Nettoyage appliqué

| Problème constaté                                                  | Décision                                           | Justification                                                              |
| -------------------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------- |
| 5 268 doublons exacts                                                | Suppression                                         | Lignes saisies deux fois, elles gonfleraient fréquence et montant         |
| 10 587 quantités négatives (factures`C…` = annulations/retours) | Suppression (`Quantity > 0`)                      | On mesure ici les achats ; les retours fausseraient le montant             |
| 2 512 prix unitaires ≤ 0 (ex. « Adjust bad debt », ajustements)   | Suppression (`UnitPrice > 0`)                     | Lignes comptables, pas des achats clients                                  |
| 132 186 lignes sans`CustomerID`                                    | Suppression                                         | Impossible de rattacher ces lignes à un client, donc de construire un RFM |
| Montant de ligne absent                                              | Création de`TotalAmount = Quantity × UnitPrice` | Nécessaire pour calculer le montant dépensé                             |

**Après nettoyage :** 392 692 lignes, **4 338 clients**, 37 pays.

---

## 🗂️ Structure du projet

```
.
├── data/
│   └── data.xlsx                          # Dataset Online Retail
├── notebooks/
│   ├── customer-segmentation_brief.ipynb  # Énoncé / brief du projet
│   └── data_exploration.ipynb             # Notebook principal (analyse complète)
├── models/
│   └── customer_segmentation_model.pkl    # Modèle sauvegardé (généré par le notebook)
├── app.py                                 # Application Streamlit
├── requirements.txt
└── README.md
```

> Le notebook lit les données via `../data/data.xlsx` : il doit donc être exécuté depuis le dossier `notebooks/`. Le dossier `models/` est créé automatiquement à la racine du projet.

---

## ⚙️ Installation

```bash
# 1. Cloner le dépôt puis se placer dans le dossier
cd customer-segmentation

# 2. (Recommandé) créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

**`requirements.txt` suggéré :**

```
pandas
numpy
matplotlib
seaborn
scikit-learn
openpyxl        # lecture du fichier .xlsx
joblib
streamlit
jupyter
```

### Lancer le notebook

```bash
cd notebooks
jupyter notebook data_exploration.ipynb
```

Exécuter toutes les cellules : le dernier bloc génère `models/customer_segmentation_model.pkl`.

---

## 🔬 Méthodologie

### 1. Construction du RFM

Une ligne par client, calculée à partir des transactions nettoyées. La date de référence est la **dernière date du dataset** (9 décembre 2011).

| Indicateur          | Définition                                            | Intérêt                                                     |
| ------------------- | ------------------------------------------------------ | ------------------------------------------------------------- |
| **Recence**   | Nombre de jours depuis le dernier achat                | Un client récent est plus actif et plus facile à réengager |
| **Frequence** | Nombre de factures distinctes (`nunique(InvoiceNo)`) | Mesure la fidélité et la régularité d'achat               |
| **Montant**   | Somme des`TotalAmount`                               | Mesure la valeur économique du client                        |

### 2. Préparation pour le clustering

Les distributions sont **très asymétriques** (skewness : Recence ≈ 1,2 ; Frequence ≈ 12,1 ; Montant ≈ 19,3), avec quelques clients extrêmes (jusqu'à 209 commandes et 280 206 £ dépensées). K-Means, basé sur des distances euclidiennes, y est très sensible. Deux transformations sont donc appliquées :

1. **`np.log1p`** : réduit l'asymétrie et l'influence des valeurs extrêmes.
2. **`StandardScaler`** : met les trois variables à la même échelle (moyenne 0, écart-type 1).

### 3. Recherche du nombre de clusters

Testé pour K de 2 à 9 :

| K           | Inertie  | Silhouette      |
| ----------- | -------- | --------------- |
| **2** | 6 502,63 | **0,436** |
| 3           | 4 893,25 | 0,339           |
| 4           | 3 988,39 | 0,338           |
| 5           | 3 371,92 | 0,314           |
| 6           | 2 901,12 | 0,310           |
| 7           | 2 632,14 | 0,298           |
| 8           | 2 444,30 | 0,285           |
| 9           | 2 207,37 | 0,271           |

**Indication algorithmique :** le meilleur coefficient de silhouette est obtenu pour **K = 2** (séparation « actifs / inactifs » la plus nette). Le coude de l'inertie reste progressif, sans cassure franche.

**Contrainte métier :** l'entreprise demande **exactement 3 classes de consommateurs**. On retient donc **K = 3**, qui reste un compromis acceptable (silhouette de 0,339, quasi identique à K = 4) et offre une segmentation plus exploitable commercialement (forte valeur / intermédiaires / inactifs). Le choix final de K combine donc l'analyse des données **et** le besoin métier.

### 4. K-Means (K = 3)

`KMeans(n_clusters=3, random_state=42, n_init=10)` entraîné sur les données log-transformées et standardisées. Les noms de segments sont attribués **automatiquement** par un score combinant les rangs de Montant, Frequence et (inverse de) Recence.

### 5. Visualisation PCA

Une PCA en 2 composantes est utilisée **uniquement pour visualiser** les segments (elle n'intervient pas dans le clustering). PC1 explique ≈ **75,1 %** de la variance et PC2 ≈ **18,8 %**, soit ≈ **93,9 %** au total : la représentation 2D est donc fidèle.

### 6. Classification Random Forest

- **X** : `Recence`, `Frequence`, `Montant` (valeurs brutes, les arbres n'ont pas besoin de normalisation)
- **y** : le cluster K-Means
- Séparation **80 % / 20 %**, **stratifiée** pour conserver les proportions de classes
- `RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)`
- Validation croisée **stratifiée à 5 folds** sur le jeu d'entraînement, puis évaluation finale sur le jeu de test

---

## 📊 Résultats

> Valeurs obtenues en exécutant le pipeline avec `random_state=42`. Un changement de version de bibliothèque peut entraîner de légères variations.

### Profil des segments

| Segment                                     | Clients | Part des clients | Récence moy. (jours) | Fréquence moy. | Montant moy. | Contribution au CA |
| ------------------------------------------- | ------- | ---------------- | --------------------- | --------------- | ------------ | ------------------ |
| 🌟**Clients à forte valeur**         | 729     | 16,8 %           | 12,9                  | 13,7            | 8 016,8      | **65,8 %**   |
| 🔄**Clients intermédiaires**         | 1 688   | 38,9 %           | 46,3                  | 3,6             | 1 402,1      | 26,6 %             |
| 💤**Clients occasionnels / inactifs** | 1 921   | 44,3 %           | 161,2                 | 1,3             | 352,0        | 7,6 %              |

**Interprétation métier**

- **Forte valeur** : achètent très récemment, très souvent et pour de gros montants. Moins de 17 % des clients génèrent près des deux tiers du chiffre d'affaires → à fidéliser en priorité (programme VIP, offres exclusives).
- **Intermédiaires** : clients encore actifs, achats modérés. Ils ont un potentiel de montée en gamme (cross-sell, offres de réachat).
- **Occasionnels / inactifs** : dernier achat ancien (~5 mois), environ une seule commande. Cibles de campagnes de réactivation à coût maîtrisé.

> Les numéros de cluster (0, 1, 2) sont arbitraires ; les **noms de segments** sont déduits des profils RFM.

### Performance du Random Forest

**Validation croisée 5 folds (entraînement)**

| Mesure            | Valeur                                         |
| ----------------- | ---------------------------------------------- |
| Accuracy par fold | 0,9755 · 0,9856 · 0,9798 · 0,9798 · 0,9712 |
| Moyenne           | **0,978**                                |
| Écart-type       | 0,005                                          |

**Jeu de test (868 clients)**

| Métrique (moyenne pondérée) | Score |
| ------------------------------ | ----- |
| Accuracy                       | 0,976 |
| Precision                      | 0,976 |
| Recall                         | 0,976 |
| F1-score                       | 0,976 |

**Matrice de confusion (lignes = réel, colonnes = prédit, clusters 0 / 1 / 2)**

|                   | Pred 0 | Pred 1 | Pred 2 |
| ----------------- | ------ | ------ | ------ |
| **Réel 0** | 330    | 3      | 5      |
| **Réel 1** | 4      | 142    | 0      |
| **Réel 2** | 9      | 0      | 375    |

Les erreurs ne concernent presque que des clients situés à la **frontière entre deux segments voisins** (aucune confusion entre « forte valeur » et « inactifs »).

> ⚠️ **Point d'attention.** Les classes à prédire proviennent **directement de K-Means**. Un score élevé signifie que le Random Forest **reproduit les règles de segmentation** du clustering ; il ne prédit en aucun cas un comportement futur du client.

### Importance des variables

| Variable  | Importance |
| --------- | ---------- |
| Frequence | 0,361      |
| Montant   | 0,343      |
| Recence   | 0,295      |

Les trois variables contribuent de façon assez équilibrée. La fréquence et le montant sont légèrement plus discriminants, ce qui est cohérent : ils sont fortement corrélés entre eux et séparent bien les clients à forte valeur des autres.

### Exemple de prédiction

Nouveau client : `Recence = 30`, `Frequence = 5`, `Montant = 500.0`
→ **Cluster 0 — Clients intermédiaires**

---

## 🖥️ Application Streamlit

L'application permet de saisir les trois valeurs RFM d'un nouveau client et affiche :

- le **cluster prédit** ;
- le **nom du segment** ;
- les **caractéristiques RFM saisies**.

### Lancement

```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`. Le fichier `models/customer_segmentation_model.pkl` doit exister (exécuter d'abord le notebook).

### Exemple minimal de `app.py`

```python
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Segmentation client", page_icon="🧩")

@st.cache_resource
def load_bundle():
    return joblib.load("models/customer_segmentation_model.pkl")

bundle = load_bundle()
model = bundle["rf_model"]
segment_names = bundle["segment_names"]

st.title("🧩 Segmentation d'un nouveau client")
st.write("Renseignez les indicateurs RFM du client pour connaître son segment.")

recence = st.number_input("Récence (jours depuis le dernier achat)", min_value=0, value=30, step=1)
frequence = st.number_input("Fréquence (nombre de commandes)", min_value=1, value=5, step=1)
montant = st.number_input("Montant total dépensé", min_value=0.0, value=500.0, step=50.0)

if st.button("Prédire le segment"):
    client = pd.DataFrame({"Recence": [recence], "Frequence": [frequence], "Montant": [montant]})
    cluster = int(model.predict(client)[0])
    st.success(f"Segment : **{segment_names[cluster]}** (cluster {cluster})")
    st.dataframe(client)
```

---

## 💾 Utiliser le modèle sauvegardé

Le fichier `.pkl` est un dictionnaire (`joblib`) contenant :

| Clé                | Contenu                                                       |
| ------------------- | ------------------------------------------------------------- |
| `rf_model`        | Random Forest entraîné                                      |
| `kmeans_model`    | K-Means final                                                 |
| `scaler`          | `StandardScaler` ajusté sur les données log-transformées |
| `segment_names`   | Correspondance cluster → nom du segment                      |
| `profile`         | Profil RFM moyen par cluster                                  |
| `cluster_profile` | Tableau de profiling complet (clients, CA, contribution…)    |
| `features`        | Liste des variables :`["Recence", "Frequence", "Montant"]`  |

```python
import joblib, pandas as pd

bundle = joblib.load("models/customer_segmentation_model.pkl")
client = pd.DataFrame({"Recence": [30], "Frequence": [5], "Montant": [500.0]})

cluster = int(bundle["rf_model"].predict(client)[0])
print(cluster, bundle["segment_names"][cluster])
```

---

## 🚧 Limites et pistes d'amélioration

- **Choix de K contraint** : le silhouette favorise K = 2 ; K = 3 est imposé par le besoin métier.
- **Retours et annulations exclus** : un client qui retourne beaucoup de produits n'est pas distingué. On pourrait déduire les retours du montant plutôt que les supprimer.
- **Clients sans `CustomerID` exclus** (~25 % des lignes), ce qui peut introduire un biais.
- **Dépendance au K-Means** : le Random Forest imite le clustering ; si la segmentation évolue, il faut réentraîner tout le pipeline.
- **Segmentation figée dans le temps** : la récence dépend de la date de référence. En production, il faudrait recalculer le RFM régulièrement.
- **Pistes** : tester d'autres algorithmes (GMM, clustering hiérarchique), ajouter des variables (panier moyen, pays, diversité produits), versionner le modèle et ajouter des tests sur l'application.

---

## 🛠️ Technologies

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib · Streamlit · Jupyter
