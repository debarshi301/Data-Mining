#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 05:33:25 2019

@author: debarshi

Name- Debarshi Dutta
Assignment- Homework 1 
CISC 6930
"""

#Question-3

import numpy as np
import matplotlib.pyplot as plt
import csv
import random

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

def plotMSE(MSE_test, title, subplotIndex):
    plt.subplot(subplotIndex)
    plt.title(title)
    plt.xlabel ('Training data')
    plt.ylabel('MSE Values')

#function to add 1's as the fiest column for the input X
def addOne(Xdata):
    X = np.insert(Xdata,0,1,axis=1)
    
    return X

#create 50(1000)_100, 100(1000)_100, 150(1000)_100

ds_1000_100 = ('train-1000-100.csv', 'test-1000-100.csv', 100, 'train-1000-100 vs test-1000-100', \
               'train-1000-100', 'test-1000-100', 'red','yellow','252')
'''ds_50_1000_10 = ('train-50(1000)-100.csv', 'test-1000-100.csv',100, 'train-50(1000)-100 vs test-1000-100', 'train-50(1000)-100', 'test-1000-100', 'red','yellow','253')
ds_100_1000_10 = ('train-100(1000)-100.csv', 'test-1000-100.csv',100, 'train-100(1000)-100 vs test-1000-100', 'train-100(1000)-100', 'test-1000-100', 'red','yellow','254')
ds_150_1000_10 = ('train-150(1000)-100.csv', 'test-1000-100.csv',100, 'train-150(1000)-100 vs test-1000-100', 'train-150(1000)-100', 'test-1000-100', 'red','yellow','255')'''

ds_list_complete =[ds_1000_100] #make them into one list
#print (ds_list_complete)

runs = 10
randomSample = 100
subplotIndex = 231

for counter in range(len(ds_list_complete)):
    train_data = np.genfromtxt( ds_list_complete[counter][0], delimiter=',', skip_header=1 ) #extracting the training data
    test_data = np.genfromtxt( ds_list_complete[counter][1], delimiter=',', skip_header=1 )#extracting the testing data
    
    xTestData = np.asmatrix( test_data[:, range(0,ds_list_complete[counter][2])] )
    xTestData_full = addOne( xTestData )
    yTestData = np.asmatrix( test_data[:, [ds_list_complete[counter][2]]] )
    
    lambda_list = list([1,25,150])
    
    for lambda_value in range(len(lambda_list)):
       
        MSEListSum = []
        
        for j in range(0,runs):
            random_list = random.sample(range(1,1000),randomSample) #generate a random list with a max\
                                                                   # value of 1000
            random_list.sort()
            #print (random_list)
            
            MSE_List = []
            
            for n in range(len(random_list)): #generate a list of random numbers
                trainData = train_data[:random_list[n]]
                xTrain = np.asmatrix( trainData[:, range(0,ds_list_complete[counter][2])] )
                xTrainData = addOne(xTrain)
                yTrainData = np.asmatrix( trainData[:, [ds_list_complete[counter][2]]] )
                
                #print(yTrainData)
                w = wList(xTrainData,yTrainData,lambda_list[lambda_value])
                mean_squared_error = MSE(xTrainData,yTrainData,w)
                
                MSE_List.append(mean_squared_error)
            
            #print (mean_squared_error)
            MSEListSum += MSE_List
            #print(MSEListSum)
            
        mseArrayReshape = np.reshape( np.array(MSEListSum) , (runs, randomSample) )
        avgMSEArray = np.divide( mseArrayReshape.sum(axis=0) , runs ) 
        
        #print(lambda_list[lambda_value])
        #print(avgMSEArray)
        plotMSE(avgMSEArray,'ℷ= ' +str(lambda_list[lambda_value]), subplotIndex)
        subplotIndex += 1

plt.subplots_adjust(wspace = 1)
plt.show()