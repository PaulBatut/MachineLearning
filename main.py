import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn import preprocessing, decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Defined 3 points in 2D-space:
X=np.array([[2, 1, 0],[4, 3, 0]])
# Calculate the covariance matrix:
X_transpose = X.transpose()
# print(X_transpose)
R=(1/3)*np.matmul(X,X_transpose)

# print(R)
# Calculate the SVD decomposition and new basis vectors:
[U,D,V]=np.linalg.svd(R)  # call SVD decomposition
u1=U[:,0] # new basis vectors
u2=U[:,1]


# Calculate the coordinates in new orthonormal basis:
Xi1 = np.matmul(np.transpose(X), u1)
Xi2 = np.matmul(np.transpose(X), u2)
# print(Xi1)
# print(Xi2)

Xrprox = np.matmul(u1[:,None], Xi1[None,:])
print(Xrprox)
# Calculate the approximation of the original from new basis
#print(Xi1[:,None]) # add second dimention to array and test it


# Check that you got the original


# Load Iris dataset as in the last PC lab:

iris = load_iris()
iris.feature_names
print(iris.feature_names)
print(iris.data[0:5, :])
print(iris.target[:])


# We have 4 dimensions of data, plot the first three colums in 3D
X=iris.data
y=iris.target
axes1=plt.axes(projection='3d')
axes1.scatter3D(X[y==0,1],X[y==0,1],X[y==0,2],color='green')
axes1.scatter3D(X[y==1,1],X[y==1,1],X[y==1,2],color='blue')
axes1.scatter3D(X[y==2,1],X[y==2,1],X[y==2,2],color='magenta')
plt.show

# Pre-processing is an important step, you can try either StandardScaler (zero mean, unit variance of features)
# or MinMaxScaler (to interval from 0 to 1)
Xscaler = StandardScaler()
Xpp=Xscaler.fit_transform(X)

# define PCA object (three components), fit and transform the data
pca = decomposition.PCA(n_components=3)
pca.fit(Xpp)
Xpca = pca.transform(Xpp)
print(pca.get_covariance())
# you can plot the transformed feature space in 3D:
axes2=plt.axes(projection='3d')
axes2.scatter3D(Xpca[y==0,0],Xpca[y==0,1],Xpca[y==0,2],color='green')
axes2.scatter3D(Xpca[y==1,0],Xpca[y==1,1],Xpca[y==1,2],color='blue')
axes2.scatter3D(Xpca[y==2,0],Xpca[y==2,1],Xpca[y==2,2],color='magenta')
plt.show()

# Import train_test_split as in last PC lab, split X (original) into train and test, train KNN classifier on full 4-dimensional X

x_train, x_test, y_train_y_test = train_test_split(X,y, test_size=0.3)
print(x_train.shape)
knn1=KNeighborsClassifier(n_neighbors = 3)
knn1.fit(X,X)
Ypred=knn1.predict()
# Import and show confusion matrix
confusion_matrix(,)
ConfusionMatrixDisplay.from_predictions(,)



# Now do the same (data set split, KNN, confusion matrix), but for PCA-transformed data (1st two principal components, i.e., first two columns).
# Compare the results with full dataset

# Now do the same, but use only 2-dimensional data of original X (first two columns)
X_trainirong, X_testirong, y_trainirong, y_testirong = train_test_split(X[:,0:1], y, test_size=0.3)
knn1 = KNeighborsClassifier(n_neighbors=3)
knn1.fit(X_trainirong,y_trainirong)
Ypredirong = knn1.predict(X_testirong)
#Import and show confusion matrix
ConfusionMatrixDisplay.from_predictions(y_testirong,Ypredirong)