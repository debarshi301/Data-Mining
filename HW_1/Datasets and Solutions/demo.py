#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:06:27 2020

@author: debarshi
"""

import csv

with open('train-1000-100.csv') as csvfile:
    reader = csv.reader(csvfile)
    for i,row in enumerate(reader):
        print(row)
        if(i >= 50):
            break
