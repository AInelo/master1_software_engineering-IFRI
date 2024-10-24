
# L'Algorithme K-Nearest Neighbors (KNN)

## Qu'est-ce que KNN ?

L'algorithme **K-Nearest Neighbors (KNN)** est une méthode utilisée en apprentissage supervisé pour résoudre des problèmes de **classification** et parfois de **régression**. Il est appelé "K-voisins les plus proches" car il se base sur les **voisins** les plus proches pour faire des prédictions.

## Comment fonctionne KNN ?

1. **Stocker les données d'entraînement** :
   - KNN ne construit pas de modèle explicite comme d'autres algorithmes. Il **mémorise simplement** les données d'entraînement, c'est-à-dire les exemples pour lesquels on connaît la réponse correcte.

2. **Calculer les distances** :
   - Lorsqu'on veut faire une prédiction pour un nouveau point, KNN mesure la distance entre ce point et tous les points d'entraînement.
   - Cette distance est généralement calculée en utilisant la **distance euclidienne**. Voici la formule :
     \[
     d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}
     \]
   - Cela calcule la distance entre deux points \(p\) et \(q\), où \(p_i\) et \(q_i\) sont les valeurs des caractéristiques (features).

3. **Trouver les k voisins les plus proches** :
   - Une fois les distances calculées, KNN trouve les **k points d'entraînement** les plus proches du nouveau point.
   - `k` est un paramètre que vous pouvez choisir. Si `k=3`, KNN choisira les 3 voisins les plus proches.

4. **Classification (vote)** :
   - Si vous utilisez KNN pour un problème de classification (par exemple, décider si une fleur est de type A ou B), KNN regarde les classes des `k` voisins et fait un **vote majoritaire**.

5. **Régression (moyenne)** :
   - Pour une tâche de régression, KNN prend la **moyenne des valeurs** des `k` voisins.

## Paramètres importants de KNN

### 1. Choix de k :
   - `k` est le **nombre de voisins** à prendre en compte.
   - Un **k trop petit** peut rendre l'algorithme trop sensible au bruit.
   - Un **k trop grand** peut inclure trop de voisins lointains et diluer la précision.

### 2. Distance :
   - **Distance Euclidienne** : Elle calcule une distance droite entre deux points.
   - **Distance de Manhattan** : Elle mesure la distance en "lignes droites", comme dans une grille de rues.

### 3. Pondération des voisins :
   - On peut donner plus d'importance aux voisins **plus proches** en les pondérant plus fortement.

## Avantages et inconvénients de KNN

### Avantages :
- **Simple à comprendre** et à implémenter.
- **Non-paramétrique**, donc aucune hypothèse sur la distribution des données.
- Fonctionne bien avec des petites données.

### Inconvénients :
- **Lent** pour prédire avec de grands ensembles de données, car il doit mesurer la distance avec chaque point.
- **Sensibilité à la dimensionnalité** : Quand il y a trop de caractéristiques, les distances deviennent moins significatives.

## Exemple en Python avec Scikit-learn

Voici comment utiliser KNN en Python avec la bibliothèque Scikit-learn :

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Chargement d'un jeu de données
iris = load_iris()
X = iris.data
y = iris.target

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création d'un modèle KNN avec k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Entraînement du modèle
knn.fit(X_train, y_train)

# Prédiction
y_pred = knn.predict(X_test)

# Évaluation
print(f"Précision : {accuracy_score(y_test, y_pred)}")
```

### Explication du code :
- **KNeighborsClassifier(n_neighbors=3)** : Crée un modèle KNN où `k=3`.
- **fit(X_train, y_train)** : Entraîne le modèle sur les données d'entraînement.
- **predict(X_test)** : Prédit les classes des nouvelles données.
- **accuracy_score** : Mesure la précision de la prédiction (pourcentage de prédictions correctes).

## Conclusion

KNN est un algorithme simple et puissant, mais il peut être lent pour les grandes données. Il nécessite un bon choix de `k` et une normalisation des données pour de meilleurs résultats.
