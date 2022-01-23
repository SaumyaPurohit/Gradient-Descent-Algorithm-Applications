#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import math


def generateXvector(X):
    vectorX = np.c_[np.ones((len(X), 1)), X]
    return vectorX

def theta_init(X):
    theta = np.random.randn(len(X[0]) + 1, 1)
    return theta

def sigmoid_function(X):
    return 1/(1 + math.e**(-X))

def Logistics_Regression(X, y, learningrate, iterations):
    y_new = np.reshape(y, (len(y), 1))
    cost_lst = []
    vectorX = generateXvector(X)
    theta = theta_init(X)
    m = len(X)
    for i in range(iterations):
        gradients = (2/m) * (vectorX.T.dot(sigmoid_function(vectorX.dot(theta)) - y_new))
        theta = theta - learningrate * gradients
        y_pred = sigmoid_function(vectorX.dot(theta))
        cost_value  = - np.sum(np.dot(y_new.T,np.log(y_pred)) + np.dot((1-y_new).T,np.log(1-y_pred))) / (len(y_pred))
        cost_lst.append(cost_value)
    plt.plot(np.arange(1, iterations), cost_lst[1:], color = "red")
    plt.title("Cost Function Graph")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    return theta
    
def column(matrix, i):
    return [row[i] for row in matrix]

def accuracy_LR(X, y, learningrate, iteration, X_test, y_test):
    ideal = Logistics_Regression(X, y, learningrate, iteration)
    hypo_line = ideal[0]
    for i in range(1, len(ideal)):
        hypo_line = hypo_line + ideal[i] * column(X_test, i - 1)
    logistic_function = sigmoid_function(hypo_line)
    for i in range(len(logistic_function)):
        if logistic_function[i] >= 0.5:
            logistic_function[i] = 1
        else:
            logistic_function[i] = 0
    last1 = np.concatenate((logistic_function.reshape(len(logistic_function), 1), y_test.reshape(len(y_test), 1)), 1)
    count = 0
    for i in range(len(y_test)):
        if last1[i][0] == last1[i][1]:
            count = count + 1
    acc = count/(len(y_test))
    return acc
        
if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris["data"]
    y = (iris["target"] == 0).astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = LogisticRegression(random_state = 0, penalty = "none")
    classifier.fit(X_train, y_train)
    print(classifier.intercept_)
    print(classifier.coef_)
    y_pred = classifier.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(Logistics_Regression(X_train, y_train, 1, 1000000))
    print(accuracy_LR(X_train,y_train, 1, 1000000, X_test, y_test))
   
