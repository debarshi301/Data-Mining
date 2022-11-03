#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 05:33:25 2019

@author: debarshi

Name- Debarshi Dutta
Assignment- Homework 1 
CISC 6930
"""

#Question-2

import numpy as np
import matplotlib.pyplot as plt
import csv

#function defintion to find the w
def wList(X,y,lambda_value):
    I = np.identity(np.shape(X)[1]) #the identity matrix has the same dimensions as\
                                    #the number of columns of the X matrix
    XTX = np.dot(X.T,X)
    lambda_I = np.dot(lambda_value,I)
    term_one = np.linalg.inv(XTX + lambda_I)
    term_two = np.dot(term_one, X.T)
    w = np.dot(term_two, y)
    
    return w

#function defintion to find the mean squared error
def MSE(X,Y,w):
    Xw = np.dot(X,w)
    mean_squared_error = np.sum(np.square((Xw - Y)))/np.shape(X)[0] #since MSE = 1/N (wX-Y)^2
    
    return mean_squared_error

def plotlambdaMSE(MSE_training, MSE_test, lambdalist, title, train_legend, text_legend,train_color, test_color, subplotIndex):
    plt.subplot(subplotIndex)
    plt.plot(lambdalist, MSE_training)
    plt.plot(lambdalist, MSE_test)
    plt.title(title)
    plt.gca().set_color_cycle([train_legend, text_legend])
    plt.legend([train_legend, text_legend])
    plt.xlabel ('Lambda(ℷ)')
    plt.ylabel('MSE Values')

#function to add 1's as the fiest column for the input X
def addOne(Xdata):
    X = np.insert(Xdata,0,1,axis=1)
    
    return X

#create 50(1000)_100, 100(1000)_100, 150(1000)_100

#lists for individual datasets
#ds_100_10 = ('train-100-10.csv', 'test-100-10.csv', 10, 'train-100-10 vs test-100-10', 'train-100-10', \
#            'test-100-10', 'red','yellow','250') #250 is the subindex
ds_100_100 = ('train-100-100.csv', 'test-100-100.csv', 100, 'train-100-100 vs test-100-100', \
              'train-100-100', 'test-100-100', 'red','yellow','251')
#ds_1000_100 = ('train-1000-100.csv', 'test-1000-100.csv', 100, 'train-1000-100 vs test-1000-100', \
#               'train-1000-100', 'test-1000-100', 'red','yellow','252')
'''ds_50_1000_10 = ('train-50(1000)-100.csv', 'test-1000-100.csv',100, 'train-50(1000)-100 vs test-1000-100', 'train-50(1000)-100', 'test-1000-100', 'red','yellow','253')
ds_100_1000_10 = ('train-100(1000)-100.csv', 'test-1000-100.csv',100, 'train-100(1000)-100 vs test-1000-100', 'train-100(1000)-100', 'test-1000-100', 'red','yellow','254')
ds_150_1000_10 = ('train-150(1000)-100.csv', 'test-1000-100.csv',100, 'train-150(1000)-100 vs test-1000-100', 'train-150(1000)-100', 'test-1000-100', 'red','yellow','255')'''

ds_list_complete =[ds_100_100] #make them into one list
#print (ds_list_complete)

folds = 10

for counter in range(len(ds_list_complete)):
    train_data = np.genfromtxt( ds_list_complete[counter][0], delimiter=',', skip_header=1 ) #extracting the training data
    test_data = np.genfromtxt( ds_list_complete[counter][1], delimiter=',', skip_header=1 )#extracting the testing data

    X_train = np.asmatrix( train_data[:, range(0,ds_list_complete[counter][2])] ) #X_training datasets
    X_train_data = addOne (X_train) #adding 1's to the first column of all training datasets
    Y_train_data = np.asmatrix(train_data[:, [ds_list_complete[counter][2]]] ) #Y_training datasets
    
    X_test = np.asmatrix(test_data[:, range(0,ds_list_complete[counter][2])] ) #X_training datasets
    X_test_data = addOne (X_test) #adding 1's to the first column of all test datasets
    Y_test_data = np.asmatrix(test_data[:, [ds_list_complete[counter][2]]] ) #Y_test datasets
    #print (X_test_data)
    #print (Y_test_data)
    
    #split the data into 10 matrix for the 10 fold CV
    training_data_split = np.split(train_data,folds)
    print (np.shape(training_data_split))
    
    
    lambda_list = list(range(0,151))
    
    for count in range(len[lambda_list]):
        
        