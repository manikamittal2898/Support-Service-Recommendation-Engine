    
""" #Using username, fetch customer ID,gender, age, buying capacity, purpose from User database
    cust_id=12224
    gender='M'
    age=35
    buy_cap=4
    purp='Academic'
    
    #Using customer id, fetch all product ids matching that customer id in final database table
    #For each product id, fetch the product type from product database table
    prod_type='Laptop for Home'
    #now for each product type, use the product type, cust_id,gender,age,buy_cap,purpose, to get
    #predicted value of services
    X_test=[purp,gender,age,max_cap,prod_type]"""




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
from sklearn.metrics import confusion_matrix

#Dividing the data into test and train sets
def train_test_split(file, model):
    X = file.iloc[1:200, :-1].values
    Y = file.iloc[1:200, -1].values
    #P=[]
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.30, random_state=0)
    
# =============================================================================
#     i= 0
#     print(X_test[i])
#     print(X_train[i])
# =============================================================================
    if model==0:
        P= naive_bayes(X_train, Y_train, X_test, Y_test)
        
    elif model==1:
        P= svm(X_train, Y_train, X_test, Y_test)
        
    elif model==2:
        P= knearestNeighbors(X_train, Y_train, X_test, Y_test)
        
    elif model==3:
        P= dec_tree(X_train, Y_train, X_test, Y_test)
        
    else:
        P= rand_forest(X_train, Y_train, X_test, Y_test)

    return P

#Model Fitting
#Model 1: Naive Bayes
def naive_bayes(X_train, Y_train, X_test, Y_test): 
    classifier = MultinomialNB()
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    acc= accuracy_score(Y_test,y_pred)
    acc=acc*100
    Accuracies.append(acc)
    print(acc)
    print("\n")
    return y_pred

#Model 2: Support Vector Machines
def svm(X_train, Y_train, X_test, Y_test):
    classifier= SVC(kernel = 'poly', gamma='scale', random_state=0)
    classifier = SVC()
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    acc= accuracy_score(Y_test,y_pred)
    acc=acc*100
    Accuracies.append(acc)
    print(acc)
    print("\n")
    return y_pred


#Model 3: K Nearest Neighbors
def knearestNeighbors(X_train, Y_train, X_test, Y_test):
    classifier= KNeighborsClassifier(n_neighbors =5, metric= 'minkowski', p=2)
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    acc= accuracy_score(Y_test,y_pred)
    acc=acc*100
    Accuracies.append(acc)
    print(acc)
    print("\n")
    return y_pred


#Model 4: Decision Tree 
def dec_tree(X_train, Y_train, X_test, Y_test): 
    classifier = DecisionTreeClassifier(max_depth=3)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    acc= accuracy_score(Y_test,y_pred)
    acc=acc*100
    Accuracies.append(acc)
    print(acc)
    print("\n")
    return y_pred
 
#Model 5: Random Forest    
def rand_forest(X_train, Y_train, X_test, Y_test): 
    classifier = RandomForestClassifier(n_estimators = 10)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)#this is the service predicted
    acc= accuracy_score(Y_test,y_pred)
    acc=acc*100
    Accuracies.append(acc)
    print(acc)
    print("\n")
    return y_pred

#Calling Methods
Predictions=[]
Accuracies=[]
List_final=[]
Model= ["Naive Bayes", "SVM", "K- Nearest Neighbors", "Decision Trees", "Random Forest"]
for i in range(0, 5):
    #Add proper file name
    my_file= pd.read_csv("Encoded_data.csv")
    print("Accuracy using "+Model[i]+" is: ")
    #train_test_split(my_file, i)
    Predictions.append(train_test_split(my_file, i))
for i in range(0, 5):
    Final_Predictions={"Model Name": None, "Accuracy": None, "Prediction": None}
    Final_Predictions["Model Name"]= Model[i]
    Final_Predictions["Accuracy"]= Accuracies[i]
    Final_Predictions["Prediction"]= Predictions[i]
    List_final.append(Final_Predictions)

List_final= sorted(List_final, key = lambda i: i['Accuracy'],reverse=True)
