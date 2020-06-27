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
#CODE
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder() 
file2= pd.read_csv(r'/home/shivam/Desktop/Final Project/flaskblog/v2.csv')
file1= pd.read_csv(r'/home/shivam/Desktop/Final Project/flaskblog/v1.csv')

Purpose = file1.iloc[:, 3].values
Gender = file1.iloc[:, 4].values
Prod_Type = file2.iloc[:, 2].values
Age= file1.iloc[:, 5].values
Max_cap= file1.iloc[:, 6].values
Service_Type= file2.iloc[:, 3].values

Purpose1= label_encoder.fit_transform(Purpose)
Gender1= label_encoder.fit_transform(Gender)
Prod_type1= label_encoder.fit_transform(Prod_Type)
Service_type1= label_encoder.fit_transform(Service_Type)

with open('Encoded_data_file.csv', 'a',newline='') as csvfile:
    filewriter= csv.writer(csvfile)
    filewriter.writerow(["Purpose", "Gender", "Age", "Max_Cap", "Prod_type", "Service"])
    for i in range(0, 100):
        filewriter.writerow([Purpose1[i], Gender1[i], Age[i], Max_cap[i], Prod_type1[i], Service_type1[i]])
csvfile.close()

#CODE
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
    my_file= pd.read_csv("Encoded_data_file.csv")
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


#CODEEEEE

#To decode: use inverse_transform(self, y)[source]
L1= label_encoder.inverse_transform(Service_type1)

#mapping encoded data with the decoded data
le_name_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
print(le_name_mapping)

# =============================================================================
# #get from predictions
# suitable_service= 0
# #Getting specific service from database
# print('Predicted Service: '+le_name_mapping.get(0))
# =============================================================================

Pred= []
#Getting all predictions from all models
for i in range(0, 5):
        Pred.append([Final_Predictions['Prediction'] for Final_Predictions in List_final])
   