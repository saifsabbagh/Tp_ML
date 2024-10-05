from sklearn.metrics import r2_score
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LinearRegression

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data=pd.read_csv('D:/l3/ML/tp/apprntissage/iris.csv')  

print(data.head())
#2
#Citer les attributs
attributs = data.columns
print("Attributs:", attributs)

#3
#Nombre d'espèces et leurs noms
nombre_especes = data['variety'].nunique()#compter le nombre d'éléments uniques dans la colonne "variety" de notre DataFrame 

especes = data['variety'].unique()#obtenir la liste des éléments uniques dans la colonne "variety" de notre DataFrame
print("Nombre d'especes:", nombre_especes)
print("Especes:", especes)

#4
#Nombre d'observations par espèces
observations_par_espece = data['variety'].value_counts()#compter combien de fois chaque valeur unique (variété) apparaît dans la colonne "variety" d'un DataFrame 
print("Nombre d'observations par espèce:")
print(observations_par_espece)
#5
#Afficher plusieurs fois 5 lignes du data set
print(data.sample(5))
#6
#Type des attributs (variables)
types_attributs = data.dtypes
print("Types des attributs:")
print(types_attributs)

################################################################
#C_1
print("Question C1 :")
summary = data.describe()
print(summary)
print("###########################################################################")

#C_2
print("Question C2 :")
plt.hist(data['sepal.length'], bins=10)
plt.hist(data['sepal.width'], bins=10)
plt.hist(data['petal.length'], bins=10)
plt.hist(data['petal.width'], bins=10)

plt.show()
print("###########################################################################")

#C_3
print("Question C3 :")
plt.boxplot(data['sepal.length'])
plt.boxplot(data['sepal.width'])
plt.boxplot(data['petal.length'])
plt.boxplot(data['petal.width'])

plt.show()
print("###########################################################################")
#c4
print("Question C4 :")

x=data[['sepal.width']]
y=data[['sepal.length']]
plt.scatter(x,y)
x=data[['petal.width']]
y=data[['petal.length']]
plt.scatter(x,y)
plt.show()

print("######################################")



#D
# Transformation des étiquettes en numériques
label_encoder = LabelEncoder()
data['variety'] = label_encoder.fit_transform(data['variety'])

X = data.drop('variety', axis=1)
y = data['variety']
# Divisez les données en ensembles d'entraînement et de test (70% d'entraînement, 30% de test)
train_set_X, test_set_X, train_set_y, test_set_y = train_test_split(X, y, test_size=0.3, random_state=42)
print("Taille de l'ensemble d'entraînement X :", len(train_set_X))
print("Taille de l'ensemble de test X :", len(test_set_X))
print("Taille de l'ensemble d'entraînement y :", len(train_set_y))
print("Taille de l'ensemble de test y :", len(test_set_y),"\n")
print("######################################")

#Regresion lineaire
model = LinearRegression()
model.fit(train_set_X, train_set_y)
#Prediction
predictions = model.predict(test_set_X)
#Evaluation
mse = mean_squared_error(test_set_y, predictions)
r2 = r2_score(test_set_y, predictions)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}\n")
print("Coefficients du modèle :", model.coef_)
print("Intercept du modèle :", model.intercept_)
print("######################################")


