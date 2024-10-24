
# Évaluation des modèles de classification

L'évaluation des modèles de classification est une étape cruciale dans l'apprentissage supervisé, car elle permet de mesurer la performance du modèle et de déterminer dans quelle mesure il effectue correctement la classification des données. Voici les concepts essentiels pour comprendre cette évaluation :

## 1. Matrice de confusion
La **matrice de confusion** est un tableau qui présente les résultats réels par rapport aux résultats prédits par un modèle de classification. Elle comporte quatre éléments principaux :

|               | Prédiction : Positif | Prédiction : Négatif |
|---------------|----------------------|----------------------|
| **Réel : Positif**  | Vrai Positif (TP)       | Faux Négatif (FN)       |
| **Réel : Négatif**  | Faux Positif (FP)       | Vrai Négatif (TN)       |

- **Vrai Positif (TP)** : Le modèle a correctement prédit que la classe est positive.
- **Faux Positif (FP)** : Le modèle a prédit que la classe est positive alors qu'elle est en réalité négative (erreur de type I).
- **Faux Négatif (FN)** : Le modèle a prédit que la classe est négative alors qu'elle est en réalité positive (erreur de type II).
- **Vrai Négatif (TN)** : Le modèle a correctement prédit que la classe est négative.

## 2. Mesures d'évaluation basées sur la matrice de confusion

Les métriques couramment utilisées pour évaluer un modèle de classification sont dérivées de la matrice de confusion :

- **Précision (Accuracy)** : La proportion des prédictions correctes (positives et négatives) parmi toutes les prédictions.
  
  \
  $$\text{Précision} = \frac{TP + TN}{TP + TN + FP + FN}$$
  
  La précision peut être trompeuse si les classes sont déséquilibrées (par exemple, si une classe est beaucoup plus fréquente que l'autre).

- **Précision (Precision)** : La proportion des vraies prédictions positives parmi toutes les prédictions positives faites par le modèle.
  
  \
  $$\text{Précision} = \frac{TP}{TP + FP}$$
  
  Elle est particulièrement utile lorsque le coût d'un faux positif est élevé.

- **Rappel (Recall ou Sensibilité)** : La proportion des vrais positifs parmi tous les exemples qui sont réellement positifs.
  
  \
  $$\text{Rappel} = \frac{TP}{TP + FN}$$
  
  Le rappel est important lorsque l'on veut minimiser les faux négatifs (par exemple, en médecine pour détecter une maladie).

- **F1-score** : Une mesure combinée de la précision et du rappel, qui est utile pour obtenir un compromis entre les deux. Il s'agit de la moyenne harmonique de la précision et du rappel.
  
  \
  $$F1 = 2 \times \frac{\text{Précision} \times \text{Rappel}}{\text{Précision} + \text{Rappel}}$$
  
  Le F1-score est particulièrement utile lorsque les classes sont déséquilibrées et que l'on souhaite équilibrer la précision et le rappel.

## 3. Courbe ROC et AUC

La **courbe ROC (Receiver Operating Characteristic)** est un graphique qui montre la performance d'un modèle de classification binaire en traçant le **taux de faux positifs** (FPR) contre le **taux de vrais positifs** (TPR) à différents seuils de classification.

- **Taux de faux positifs (FPR)** : C'est le rapport entre les faux positifs et l'ensemble des négatifs réels.
  
  \
  $$FPR = \frac{FP}{FP + TN}$$

- **Taux de vrais positifs (Rappel ou TPR)** : C'est le même que le rappel.

Chaque point sur la courbe ROC représente un seuil de classification différent, et la courbe montre comment le modèle se comporte en fonction de ces seuils.

- **AUC (Area Under the Curve)** : Il s'agit de l'aire sous la courbe ROC. L'AUC est une mesure unique qui résume la performance du modèle sur tous les seuils. Plus l'AUC est proche de 1, meilleur est le modèle.

## 4. Métriques spécifiques aux classes déséquilibrées

Lorsque les classes sont fortement déséquilibrées (par exemple, 90% des données sont dans une classe, 10% dans l'autre), les métriques comme la précision peuvent être trompeuses. Dans ces cas, les métriques suivantes sont plus appropriées :

- **Matrice de confusion ajustée** : Prenez en compte le déséquilibre des classes en examinant des métriques comme la précision, le rappel et le F1-score au lieu de la précision brute.
  
- **Score Kappa de Cohen** : Une mesure qui tient compte des prédictions correctes attendues par hasard, utile pour les classes déséquilibrées.

## 5. Exemple pratique en Python

Utilisons Scikit-learn pour évaluer un modèle de classification :

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# Chargement d'un jeu de données exemple (cancer du sein)
data = load_breast_cancer()
X = data.data
y = data.target

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création et entraînement d'un modèle (forêt aléatoire)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Matrice de confusion
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))

# Rapport de classification
print("\nRapport de classification :\n", classification_report(y_test, y_pred))

# Score AUC
y_proba = model.predict_proba(X_test)[:, 1]
print("\nAUC : ", roc_auc_score(y_test, y_proba))
```

## Résultats :
- **Matrice de confusion** : Montre les vrais positifs, faux positifs, etc.
- **Rapport de classification** : Précision, rappel, F1-score pour chaque classe.
- **AUC** : Indique la qualité de la classification en fonction des seuils de décision.

## Conclusion

L'évaluation des modèles de classification permet de comprendre la performance du modèle à travers plusieurs métriques comme la précision, le rappel, et l'AUC, selon la tâche à accomplir. Ces mesures sont cruciales pour choisir le bon modèle et ajuster ses paramètres afin d'améliorer les résultats.
