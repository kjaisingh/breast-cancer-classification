#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:07:18 2019

@author: jaisi8631
"""

# dataset derived from: https://www.kaggle.com/paultimothymooney/breast-histopathology-images/downloads/breast-histopathology-images.zip/1


# imports
from zipfile import ZipFile


# unzip initial zip file
zip = ZipFile("breast-histopathology-images.zip")
zip.extractall("breast-histopathology-images")


# unzip secondary zip file
zip = ZipFile("breast-histopathology-images/IDC_regular_ps50_idx5.zip")
zip.extractall("breast-histopathology-images/IDC_regular_ps50_idx5")