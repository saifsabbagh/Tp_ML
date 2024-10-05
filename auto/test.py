from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import jaccard_score


iris=load_iris()

X_train_iris = iris.data
y_train_iris = iris.target
# k=3
for k in range(2, 11):
    kmeans= KMeans(n_clusters=k)

    kmeans.fit(X_train_iris)

    Y_P=kmeans.predict(X_train_iris)


    confusion_metrix = confusion_matrix(y_train_iris, Y_P)
    # print(confusion_metrix )
    # sns.heatmap(confusion_metrix,annot = True, cmap = "YlGnBu", fmt = "g")
    # plt.show()
    print("k=",k,":")
    print(jaccard_score(y_train_iris,Y_P,average='weighted',))



