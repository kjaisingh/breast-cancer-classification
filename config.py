#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 11:31:48 2019

@author: jaisi8631
"""

# imports
import os
 

# declare necessary constants
ORIG_INPUT_DATASET = "breast-histopathology-images/IDC_regular_ps50_idx5"
 
BASE_PATH = "datasets"
 
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
 
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

WIDTH, HEIGHT = 48, 48
NB = 2
BS = 64
EPOCHS = 10