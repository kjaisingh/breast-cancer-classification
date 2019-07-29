#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:33:33 2019

@author: jaisi8631
"""

# imports
import numpy as np
import config
import argparse
import imutils
import cv2
import os

from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD


# argument parser for test image file name
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type = str, default = "test.png",
                help = "path to test image")
args = vars(ap.parse_args())


# load and compile model
model = load_model('model.h5')
model.compile(loss = "binary_crossentropy", 
              optimizer = SGD(lr=0.001, momentum=0.9), 
              metrics=["accuracy"])


# preprocess image
image = cv2.imread(args["image"])
image = cv2.resize(image, (config.WIDTH, config.HEIGHT))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis = 0)


# make prediction for image
result = model.predict(image)
pred = np.argmax(result, axis = 1)
prediction = "UNRECOGNIZABLE"
if(pred[0] == 0):
    prediction = "Normal"
else:
    prediction = "Breast Cancer"


# print result
print("The prediction is: " + prediction)