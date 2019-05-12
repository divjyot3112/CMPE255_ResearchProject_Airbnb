#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 13:14:27 2018

@author: vinsingh
"""

import pandas as pd
import numpy as np
import re
from sklearn import datasets, linear_model
from sklearn import preprocessing,cross_validation,svm
import nltk, datetime
import sklearn.metrics as metrics
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn import metrics
from sklearn import ensemble
from sklearn import svm

calendarDF1=pd.read_csv("calendar_detail.csv")


calendarDF=pd.read_csv("calendar_detail.csv")

calendarDF.head()

#replacing NaN values with 0
calendarDF.fillna(0, inplace=True)
calendarDF = calendarDF[calendarDF.price != 0]

price = calendarDF['price']
prices=[]

for c in cost:
    c = re.sub('[^0-9.]+','', c)
    prices.append(float(c))
    
#replace the price column with the new column
calendarDF['price']=prices

calendarDF = calendarDF[calendarDF.price >= 0]

#separating date column into day month and year
calendarDF['Year'],calendarDF['Month'],calendarDF['Day']=calendarDF['date'].str.split('-',2).str
calendarDF.head()


y_df=calendarDF.groupby(['Year','Month']).price.mean()
y_df=y_df.reset_index()
y_df=y_df.rename(columns={'price':'average_Price'})
y_df['year-Month']=y_df['Year'].map(str) + "-" + y_df['Month'].map(str)
#y_df.to_csv('year_month_data.csv')
y_df.head()

## Linear Regression 
split_data= calendarDF.drop(['price','available','date','Year','Month','Day'],axis=1)
train1,test1,train2,test2=cross_validation.train_test_split(split_data,calendarDF.price, test_size=0.4,train_size = 0.6,random_state=13)

linear_reg = linear_model.LinearRegression()
linear_reg.fit(train1, train2)
linear_reg_error = metrics.median_absolute_error(test2, linear_reg.predict(test1))
print ("Linear Regression: " + str(linear_reg_error))

#SVM

model = svm.SVC()
model.fit(train1, train2)
linear_reg_error = metrics.median_absolute_error(test2, model.predict(test1))
print ("SVM Regression: " + str(linear_reg_error))

