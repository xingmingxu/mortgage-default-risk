# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 23:36:29 2020

@author: User
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

fold = ("C:\\Users\\User\\Documents\\Xingming\\Programming\\"
    + "Projects\\germanBank\\") ## insert folder
pm = pd.read_csv(fold + "germancredit.csv")
n = len(pm)

## define x, y
y = pm.iloc[:, 0]
x = pm.iloc[:, [2, 5, 8, 11, 13, 16, 18]]

## define training and testing datasets
prop = 0.2 # proportion
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = prop)

reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
reg.coef_

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(7,)))

# Add one hidden layer 
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(x_train, y_train, batch_size = 256, epochs = 20, verbose = 1)
y_prednn = model.predict(x_test)

for i in range(len(y_prednn)):
    if y_prednn[i] < 0.5:
        y_prednn[i] = 0
    else:
        y_prednn[i] = 1

confusion_matrix(y_test, y_prednn)
precision_score(y_test, y_prednn)
recall_score(y_test, y_prednn)


#Create a Gaussian Classifier
clfrf = RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clfrf.fit(x_train,y_train)
y_predrf = clfrf.predict(x_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_predrf))
confusion_matrix(y_test, y_predrf)