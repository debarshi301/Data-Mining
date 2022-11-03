#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:27:21 2019

@author: debarshi
"""

from scipy.io import arff
import pandas as pd

#import sys
#
#stdoutOrigin=sys.stdout 
#sys.stdout = open("output.txt", "w")

#loading arff file as dataframe
data_adolescent = (arff.loadarff('Autism-Adolescent-Data.arff'))
data_adult = (arff.loadarff('Autism-Adult-Data.arff'))
data_child = (arff.loadarff('Autism-Child-Data.arff'))
#print (data_adolescent)
##print (type(data_adolescent))
df_adolescent = pd.DataFrame (data_adolescent[0])
df_child = pd.DataFrame(data_child[0])
#print (df_adolescent.head(0))
#print (df_adolescent)

#counting total Nans

#print (df_child.isnull().sum())
print (df_child.isnull().sum().sum()/len(df_child.index)* 100)

#drop nan rows
#df_child_new = df_child.dropna()
#print (df_child_new)


#check for null values





##logging the output
#sys.stdout.close()
#sys.stdout=stdoutOrigin