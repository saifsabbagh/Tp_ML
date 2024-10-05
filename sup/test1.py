import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics;
import seaborn as sns
import numpy as np

 
 
dataset=pd.read_csv('D:/l3/ML/tp/sup/diabetes.csv')

p1=dataset.head(6)
print(p1)
p2=dataset.tail(6)
print(p2)
dataset.shape
dataset.info()#imprime des informations sur le DataFrame

plt.hist(dataset['Pregnancies'], bins='auto')
plt.hist(dataset['Glucose'], bins='auto')
plt.hist(dataset['BloodPressure'], bins='auto')
plt.hist(dataset['SkinThickness'], bins='auto')
plt.hist(dataset['Insulin'], bins='auto')
plt.hist(dataset['BMI'], bins='auto')
plt.hist(dataset['DiabetesPedigreeFunction'], bins='auto')
plt.hist(dataset['Age'], bins='auto')
plt.hist(dataset['Outcome'], bins='auto')

dataset.hist(figsize=(20,20))
plt.show()
print(dataset['Outcome'].value_counts())
dataset['Outcome'].hist()
plt.show()


x= dataset[["Pregnancies",	"Glucose",	"BloodPressure",	"SkinThickness",	"Insulin",	"BMI",	"DiabetesPedigreeFunction", "Age"]]
y=dataset['Outcome']

# Initialiser le scaler
scaler = StandardScaler()

# Adapter le scaler aux données d'apprentissage et normaliser les données
X_scaled = scaler.fit_transform(x)
print(X_scaled)
print(y)

###############
x_train, x_test, y_train, y_test = train_test_split(X_scaled,y,test_size = 0.3, random_state = 42)
# print(x_train, x_test, y_train, y_test)

classifier = LogisticRegression()
classifier.fit(x_train,y_train)

train_prediction = classifier.predict(x_train)
# print(train_prediction)

y_test_pred = classifier.predict(x_test)

print("#################################")


# print(accuracy_score(y_test_pred,y_test))

# print(accuracy_score(train_prediction,y_train))

cnf_matrix = metrics.confusion_matrix(y_test, y_test_pred)
print("t1",cnf_matrix)
sns.heatmap(pd.DataFrame(cnf_matrix),annot = True, cmap = "YlGnBu", fmt = "g")
# plt.plot(pd.DataFrame(cnf_matrix))
plt.show()

######


#données a votre choix
input_data =  (5,166,72,19,175,25.8,0.587,51)
#changer l'input_data en tableau numpy
input_data_as_numpy_array = np.asarray(input_data)
#remodeler le tableau comme nous me ^révoyons pour une instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#standardiser les données d'entrée
scaler = StandardScaler()
scaler.fit(input_data_reshaped)
std_data = scaler.fit_transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
if(prediction[0] == 0):
  print('La personne n"est pas diabetique')
else:
  print('La personne est diqbétique')