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
file2= pd.read_csv(r'C:\Users\padma\Desktop\Pratishtha\CS\Recommendation Engine\Trial Code\Munnu1.csv')
file1= pd.read_csv(r'C:\Users\padma\Desktop\Pratishtha\CS\Recommendation Engine\Trial Code\Munnu.csv')

Purpose = file1.iloc[:, 3].values
Gender = file1.iloc[:, 4].values
Prod_Type = file2.iloc[:, 2].values
Age= file1.iloc[:, 5].values
Max_cap= file1.iloc[:, 6].values
Service_Type= file2.iloc[:, 3].values

Purpose1= label_encoder.fit_transform(Purpose)
L1= label_encoder.inverse_transform(Purpose1)
#mapping encoded data with the decoded data
le_name_mapping1 = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
#print(le_name_mapping1)

Gender1= label_encoder.fit_transform(Gender)
L2= label_encoder.inverse_transform(Gender1)
#mapping encoded data with the decoded data
le_name_mapping2 = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
#print(le_name_mapping2)

Prod_type1= label_encoder.fit_transform(Prod_Type)
L3= label_encoder.inverse_transform(Prod_type1)
#mapping encoded data with the decoded data
le_name_mapping3 = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(le_name_mapping3)

Service_type1= label_encoder.fit_transform(Service_Type)
L4= label_encoder.inverse_transform(Service_type1)
#mapping encoded data with the decoded data
le_name_mapping4 = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
#print(le_name_mapping4)

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
# =============================================================================
# L1= label_encoder.inverse_transform(Service_type1)
# 
# #mapping encoded data with the decoded data
# le_name_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
# print(le_name_mapping)
# =============================================================================

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

file= pd.read_csv("Encoded_data_file.csv")
X_train = file.iloc[1:200, :-1].values
Y_train = file.iloc[1:200, -1].values
purp='Academic'
if purp=='Academic':
    purp= 0
elif purp=='Big Organization':
    purp= 1
elif purp=='Gaming Purpose':
    purp= 2
elif purp=='Personal Use':
    purp= 3
else:
    purp= 4

g='F'
if g== 'F':
    g= 0
else:
    g= 1
age=20
max=3
prod='Laptop for Home'
if prod==207552:
    prod= 0
if prod==386534:
    prod=1
if prod==457953:
    prod= 2
if prod==648367:
    prod= 3
if prod==763457: 
    prod= 4
else: 
    prod= 5

def get_service(prod):
#for services
    if prod==0:
        prod='Accidental Damage Service' 
    elif prod==1:
        prod= 'Base Warranty'
    elif prod==2:
        prod= 'Dell Technologies Unified Workspace'
    elif prod==3:
        prod= 'Designated Support Engineer(DSE)'
    elif prod==4:
        prod= 'FIX your PC'
    elif prod==5:
        prod= 'Keep Your Hard Drive'
    elif prod==6:
        prod= 'MyService360'
    elif prod==7:
        prod='Optimize for Storage'
    elif prod==8:
        prod= 'PCAAS for Business'
    elif prod==9:
        prod= 'PCAAS for Enterprise'
    elif prod==10:
        prod= 'Premium Support'
    elif prod==12:
        prod= 'Premium Support Plus'
    elif prod==13:
        prod= 'ProSupport'
    elif prod==14:
        prod= 'ProSupport Flex'
    elif prod==15:
        prod= 'ProSupport One for Data Center'
    elif prod==16:
        prod= 'ProSupport Plus'
    elif prod==17:
        prod= 'Secure Remote Services'
    elif prod==18:
        prod= 'Service Account Manager(SAM)'
    elif prod==19:
        prod= 'Support Assist'
    elif prod==20:
        prod= 'TechDirect'
    elif prod==21:
        prod= 'Technical Account Manager(TAM)'
    else:
        prod= 'Warranty Extension'
    
    return prod

X_test=[[purp,g,age,max,prod]]

#Model 1: Naive Bayes
def naive_bayes1(X_train, Y_train, X_test): 
    classifier = MultinomialNB()
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    return y_pred

#Model 2: Support Vector Machines
def svm1(X_train, Y_train, X_test):
    classifier= SVC(kernel = 'poly', gamma='scale', random_state=0)
    classifier = SVC()
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    return y_pred


#Model 3: K Nearest Neighbors
def knearestNeighbors1(X_train, Y_train, X_test):
    classifier= KNeighborsClassifier(n_neighbors =5, metric= 'minkowski', p=2)
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    return y_pred


#Model 4: Decision Tree 
def dec_tree1(X_train, Y_train, X_test): 
    classifier = DecisionTreeClassifier(max_depth=3)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    return y_pred
 
#Model 5: Random Forest    
def rand_forest1(X_train, Y_train, X_test): 
    classifier = RandomForestClassifier(n_estimators = 10)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)#this is the service predicted
    return y_pred


Preds= []
P1= dec_tree1(X_train, Y_train, X_test)
P1= get_service(P1)
P2= rand_forest1(X_train, Y_train, X_test)
P2= get_service(P2)
P3= knearestNeighbors1(X_train, Y_train, X_test)
P3= get_service(P3)
P4= svm1(X_train, Y_train, X_test)
P4= get_service(P4)
P5= naive_bayes1(X_train, Y_train, X_test)
P5= get_service(P5)

#le_name_mapping4.get(P1)

Preds.append(P1)
Preds.append(P2)
Preds.append(P3)
Preds.append(P4)
Preds.append(P5)
   
