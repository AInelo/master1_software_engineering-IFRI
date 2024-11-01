# -*- coding: utf-8 -*-
"""
TP_0

@author: Rath
"""
# Importing the libraries
import pandas as pd

# Importing the dataset
# dataset = pd.read_csv('C:\Users\HP\Desktop\MASTER1_SOFTWARE_ENGENIEERING-IFRI\Cours\4-Outils_de_résolution_de_problèmes_complexes_RO_et_IAs\Code_Machine_learning\By_Prof_Rateil\base3.csv')
dataset = pd.read_csv('.\base3.csv')

##KNN without normalization

### Extract input and outup data
X=dataset.iloc[:,[2,3]]
y = dataset.iloc[:, -1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)
 
#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier
#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_predc = knn.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_predc))
 
## compute the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predc)
