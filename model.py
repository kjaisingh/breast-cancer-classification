#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 11:37:41 2019

@author: jaisi8631
"""

# imports
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.applications import VGG19
from imutils import paths
from keras.models import Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

import matplotlib
import config
import matplotlib.pyplot as plt
import numpy as np


# conv net class declaration
class CancerConvNet:
    @staticmethod
    def build():
        base = VGG19(weights = "imagenet", include_top=False, 
                           input_shape = (config.WIDTH, config.HEIGHT, 3))
        x = base.output
        x = Flatten()(x)
        x = Dense(1024, activation = "relu")(x)
        x = Dropout(0.4)(x)
        x = Dense(1024, activation = "relu")(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation = "relu")(x)
        x = Dropout(0.2)(x)
        preds = Dense(config.NB, activation = "softmax")(x)
        model = Model(input = base.input, output = preds)
        return model