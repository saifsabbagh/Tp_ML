#1 Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#2 

data=pd.read_csv('Salary_Data.csv')#lire la base de données
data.head()#La méthode head() renvoie les 5 premières lignes si aucun nombre n'est spécifié
data.info()#imprime des informations sur le DataFrame
x=data[['YearsExperience']]
y=data[['Salary']]# Définir les variables explicatives X et la variable cible y
#un diagramme où chaque valeur de l'ensemble de données est représentée par un point
plt.scatter(x,y)
#3
#Utiliser la fonction train_test_split pour diviser la base de données en deux ensembles:
#un ensemble d'entraînement 
#un ensemble de test 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
#4
 # Créer un modèle de régression linéaire
regressor=LinearRegression()
# Entraîner le modèle sur l'ensemble d'entraînement
regressor.fit(x_train,y_train)
# Afficher les coefficients du modèle
print(+regressor.coef_)
# Afficher l'intercept du modèle
print(regressor.intercept_)
# Tracer la ligne de régression
ordonne=np.linspace(0,15,1000)

plt.scatter(x,y)

plt.plot(ordonne,regressor.coef_[0]*ordonne+regressor.intercept_,color='r')
# Étiqueter les axes
plt.xlabel('Years of experience')
plt.ylabel('Salary')
y_predict=regressor.predict(x_test)

# Calculer l'erreur absolue moyenne (MAE)
print('MAE:',metrics.mean_absolute_error(y_test,y_predict))
# Calculer l'erreur quadratique moyenne (MSE)
print('MSE:',metrics.mean_squared_error(y_test,y_predict))
 #Calculer la racine de l'erreur quadratique moyenne (RMSE
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
# Calculer le score R-squared
print('R²',metrics.r2_score(y_test,y_predict))

