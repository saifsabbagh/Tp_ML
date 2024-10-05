import pandas as pd
from seaborn.axisgrid import np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Lire le fichier diabetes.csv
# Pour mieux explorer notre base de données, nous souhaitons suivre quelques instructions pour analyser le jeu de données.
# Importation, exploration et visualisation du dataset.
# Télécharger et enregistrer vos données
dataset = pd.read_csv('D:/l3/ML/tp/sup/diabetes.csv')

# Afficher les 6 premières lignes de la base de données
print(dataset.head(6))

# Afficher les 6 dernières lignes de la base de données
print(dataset.tail(6))

# Afficher les informations
dataset.info()

# Afficher les histogrammes des caractéristiques en fonction de leurs apparitions
dataset.hist(figsize=(20, 20))
plt.show()

# Afficher la distribution des valeurs dans la colonne 'Outcome' et afficher son histogramme
plt.figure(figsize=(8, 6))
sns.countplot(x='Outcome', data=dataset)
plt.show()

# Séparer vos données en deux variables X et Y respectivement, les valeurs de caractéristiques et la valeur cible.
# Diviser le dataframe en composants x (variables prédictives) et y (cible : prédiction du diagnostic)
X = dataset.drop('Outcome', axis=1)
Y = dataset['Outcome']
# Pour une meilleure efficacité, normaliser les données
# Utiliser la bibliothèque
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Après avoir défini les deux variables X et Y respectivement, les valeurs de caractéristiques et la valeur cible.
# On utilise les fonctions train-test split pour fractionner l'ensemble des données en données de train et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Instanciation du modèle (choisir entre Logistic Regression et SVM)
classifier = svm.SVC()

# Ajuster le modèle en utilisant les données d'entrainement
classifier.fit(X_train, Y_train)

# Calculer l'accuracy de l'ensemble de train
train_prediction = classifier.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_prediction)
print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

# Faire des prédictions sur l'ensemble de test
test_prediction = classifier.predict(X_test)

# Calculer l'accuracy de l'ensemble de test
test_accuracy = accuracy_score(Y_test, test_prediction)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Visualiser la matrice de confusion
cnf_matrix = confusion_matrix(Y_test, test_prediction)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()