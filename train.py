#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 11:45:04 2019

@author: jaisi8631
"""

# imports
from model import CancerConvNet
import config

import matplotlib.pyplot as plt
plt.use("Agg")
import numpy as np
import argparse
import os
 
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths


# create argument parser for model output
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())


# create necessary variables 
trainPaths = list(paths.list_images(config.TRAIN_PATH))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))


# create data generators
trainAug = ImageDataGenerator(
	rescale = 1/255.0,
	rotation_range = 20,
	zoom_range = 0.05,
	width_shift_range = 0.1,
	height_shift_range = 0.1,
	shear_range = 0.05,
	horizontal_flip = True,
	vertical_flip = True,
	fill_mode = "nearest")

valAug = ImageDataGenerator(rescale = 1./255,
                            fill_mode = "nearest")


# create data augmentors
trainGen = trainAug.flow_from_directory(
	config.TRAIN_PATH,
	class_mode = "categorical",
	target_size = (config.WIDTH, config.HEIGHT),
	color_mode = "rgb",
	shuffle = True,
	batch_size = config.BS)
 
valGen = valAug.flow_from_directory(
	config.VAL_PATH,
	class_mode = "categorical",
	target_size = (config.WIDTH, config.HEIGHT),
	color_mode = "rgb",
	shuffle = False,
	batch_size = config.BS)
 
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode = "categorical",
	target_size = (config.WIDTH, config.HEIGHT),
	color_mode = "rgb",
	shuffle = False,
	batch_size = config.BS)


# build model
model = CancerConvNet.build()


# select trainable layers
for layer in model.layers[:15]:
    layer.trainable=False
for layer in model.layers[15:]:
    layer.trainable=True


# print information regarding model
print(model.summary())


# create early stopping callback
early = EarlyStopping(monitor = 'val_acc', min_delta = 0, 
                      patience = 10, verbose= 1 , mode = 'auto')


# compile model
model.compile(
        loss = "binary_crossentropy", 
        optimizer = SGD(lr=0.001, momentum=0.9),
        metrics = ["accuracy"])


# train model
H = model.fit_generator(
        trainGen,
        epochs = config.EPOCHS,
        steps_per_epoch = totalTrain // config.BS,
        validation_data = valGen,
        validation_steps = totalVal // config.BS,
        callbacks = [early])


# save model
model.save('model.h5')


# make predictions
testGen.reset()
modelPreds = model.predict_generator(testGen, 
                                     steps=(totalTest // config.BS) + 1)
modelPreds = np.argmax(modelPreds, axis=1)
print(classification_report(testGen.classes, modelPreds,
                            target_names = testGen.class_indices.keys()))


# save plot to disk
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, config.EPOCHS), H.history["acc"], label = "train_acc")
plt.plot(np.arange(0, config.EPOCHS), H.history["val_acc"], label = "val_acc")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy %")
plt.legend(loc = "lower left")
plt.savefig(args["plot"])