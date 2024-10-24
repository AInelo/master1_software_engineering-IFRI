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
