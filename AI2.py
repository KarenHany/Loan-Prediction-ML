#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:51:55 2024

@author: karimessam
"""
# import dataset

import pandas as pd
import numpy as np

#read dataset
dataset=pd.read_csv('D:\\sections\\int to AI\\ML\\data.csv')

#drop unwanted columns


dataset=dataset.drop('id',axis=1)






#split dataset into X and Y
X=dataset.iloc[:,dataset.columns!='diagnosis'].values
Y=dataset.iloc[:,0].values


from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

X=imp_mean.fit_transform(X)

#preprocessing for Y (encoding)
from sklearn.preprocessing import LabelEncoder
LabelEncoder_Y= LabelEncoder()
Y=LabelEncoder_Y.fit_transform(Y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

from sklearn import tree
DT=tree.DecisionTreeClassifier(criterion='entropy',random_state=0)
y_pred=DT.fit(x_train, y_train).predict(x_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=0, max_iter=50)
y_pred3=mlp.fit(x_train, y_train).predict(x_test)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred2 = gnb.predict(x_test)

from sklearn.metrics import accuracy_score
tree_score=accuracy_score(y_pred, y_test)
nn_score=accuracy_score(y_pred3, y_test)
naive_acc=accuracy_score(y_pred2, y_test)
tree.plot_tree(DT)
