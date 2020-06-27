#Importing Libraries
import numpy as np
import pandas as pd
import csv
import os

#importing libraries for models and calculations
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn import metrics


#Dividing the data into test and train sets
def train_test_split(file, model):
    X = file.iloc[:, :-1].values
    Y = file.iloc[:, -1].values
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.20, random_state=0)
    
    if model==0:
        naive_bayes(X_train, Y_train, X_test, Y_test)
    elif model==1:
        svm(X_train, Y_train, X_test, Y_test)
    elif model==2:
        knearestNeighbors(X_train, Y_train, X_test, Y_test)
    elif model==3:
        dec_tree(X_train, Y_train, X_test, Y_test)
    else:
        rand_forest(X_train, Y_train, X_test, Y_test)



#Model Fitting
#Model 1: Naive Bayes
def naive_bayes(X_train, Y_train, X_test, Y_test): 
    classifier = MultinomialNB()
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    acc= accuracy_score(Y_test,y_pred)
    acc=acc*100
    print(acc)
    print("\n")
   # X_test.fillna(X_test.mean())
#Model 2: Support Vector Machines
def svm(X_train, Y_train, X_test, Y_test):
    classifier= SVC(kernel = 'poly', gamma='scale', random_state=0)
    classifier = SVC()
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    acc= accuracy_score(Y_test,y_pred)
    acc=acc*100
    print(acc)
    print("\n")
    
#Model 3: K Nearest Neighbors
def knearestNeighbors(X_train, Y_train, X_test, Y_test):
    classifier= KNeighborsClassifier(n_neighbors =5, metric= 'minkowski', p=2)
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    acc= accuracy_score(Y_test,y_pred)
    acc=acc*100
    print(acc)
    print("\n")

#Model 4: Decision Tree 
def dec_tree(X_train, Y_train, X_test, Y_test): 
    classifier = DecisionTreeClassifier(max_depth=3)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    acc= accuracy_score(Y_test,y_pred)
    acc=acc*100
    print(acc)
    print("\n")

#Model 2: Random Forest    
def rand_forest(X_train, Y_train, X_test, Y_test): 
    classifier = RandomForestClassifier(n_estimators = 10)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    acc= accuracy_score(Y_test,y_pred)
    acc=acc*100
    print(acc)
    print("\n")


#Calling Methods
Model= ["Naive Bayes", "SVM", "K- Nearest Neighbors", "Decision Trees", "Random Forest"]
for i in range(0, 5):
    #Add proper file name
    my_file= pd.read_csv("Encoded_data.csv")
    print("Accuracy using "+Model[i]+" is: ")
    train_test_split(my_file, i)
    