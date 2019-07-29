#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 11:32:35 2019

@author: jaisi8631
"""

# imports
import config
from imutils import paths
import random
import shutil
import os


# create necessary variables and data frameworks
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.shuffle(imagePaths)


pivotTrain = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:pivotTrain]
testPaths = imagePaths[pivotTrain:]

pivotVal = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:pivotVal]
trainPaths = trainPaths[pivotVal:]

datasets = [
	("training", trainPaths, config.TRAIN_PATH),
	("validation", valPaths, config.VAL_PATH),
	("testing", testPaths, config.TEST_PATH)
]


# organise dataset
for (dType, imagePaths, baseOutput) in datasets:

	if not os.path.exists(baseOutput):
		os.makedirs(baseOutput)
 
	for inputPath in imagePaths:
		filename = inputPath.split(os.path.sep)[-1]
		label = filename[-5:-4]
		labelPath = os.path.sep.join([baseOutput, label])
 
		if not os.path.exists(labelPath):
			os.makedirs(labelPath)
 
		p = os.path.sep.join([labelPath, filename])
		shutil.move(inputPath, p)