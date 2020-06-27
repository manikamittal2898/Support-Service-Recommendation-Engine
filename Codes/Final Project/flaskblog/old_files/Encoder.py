import numpy as np
import pandas as pd
import csv
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

label_encoder = preprocessing.LabelEncoder() 
file1= pd.read_csv('Munnu.csv')
file2= pd.read_csv('Munnu1.csv')
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

with open('Encoded_data.csv', 'a',newline='') as csvfile:
    filewriter= csv.writer(csvfile)
    filewriter.writerow(["Purpose", "Gender", "Age", "Max_Cap", "Prod_type", "Service"])
    for i in range(0, 100):
        filewriter.writerow([Purpose1[i], Gender1[i], Age[i], Max_cap[i], Prod_type1[i], Service_type1[i]])
csvfile.close()

#To decode: use inverse_transform(self, y)[source]
# =============================================================================
# L1=[]
# for i in range(0, len(Gender1)):
#     L1.append(label_encoder.inverse_transform(Gender1[i]))
# =============================================================================
