# -*- coding: utf-8 -*-
"""
TP_1

@author: Rath
"""
# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('base3.csv')


##KNN without normalization

### Extract input and output data
X=dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
 
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


##KNN without normalization and with estimated

### Extract input and output data
X=dataset.iloc[:,[3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
 
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


##KNN with normalization 

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0, 1)) 
dataset['Age']=scaler.fit_transform(dataset[['Age']])
dataset['EstimatedSalary']=scaler.fit_transform(dataset[['EstimatedSalary']])

### Extract input and outup data
X1=dataset.iloc[:,[2,3]].values
y1 = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.3, random_state = 1)

#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier
#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(X1_train, y1_train)

#Predict the response for test dataset
y1_predc = knn.predict(X1_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y1_test, y1_predc))

## compute the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y1_test, y1_predc)


print('----------Thir Separation------------------')

##KNN with normalization with addition of Sex variable

## transform qualitative variable in binary in dataset
from pandas import get_dummies
dataset =get_dummies(dataset)
### Extract input and outup data
X2=dataset.iloc[:,[1,2,4,5]].values
y2 = dataset.iloc[:, -3].values

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0, 1)) 
dataset['Age']=scaler.fit_transform(dataset[['Age']])
dataset['EstimatedSalary']=scaler.fit_transform(dataset[['EstimatedSalary']])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.3, random_state = 1)
 

#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(X2_train, y2_train)

#Predict the response for test dataset
y2_predc = knn.predict(X2_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y2_test, y2_predc))

## compute the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y2_test, y2_predc)