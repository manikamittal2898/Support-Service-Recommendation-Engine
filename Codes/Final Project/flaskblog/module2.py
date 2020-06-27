import numpy as np
import pandas as pd
import csv
import os



dataset = pd.read_csv('/home/shivam/Desktop/Final Project/flaskblog/Final_Database.csv')
user_id=dict['id']
final_list=[]
list1=[]
for i, j in dataset.iterrows():
	if j['User ID']==user_id:
         
         X=j['Service']
         Y=j['Category']
         for index, row in dataset.iterrows():
	         if row['Category']==Y and X!=row['Service']:
		         list1.append(row['Service'])
	         elif row['Months to expire']<=5 and row['Service']==X:
		         list1.append(row['Service'])


for s in list1:
	if s not in final_list:
		final_list.append(s)
print(final_list)

