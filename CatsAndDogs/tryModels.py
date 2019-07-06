import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

#Load saved python serialized objects...
def loadData(trainX, trainY):
    trainX = pickle.load(open("X.pickle", "rb"))
    trainY = pickle.load(open("Y.pickle", "rb"))
    return (trainX, trainY)

imgSize = 128
denseLayers = [0, 1, 2]
layerSizes = [32, 64, 128]
conv2DLayers = [1, 2, 3]

trainX = []
trainY = []   
(trainX, trainY) = loadData(trainX, trainY)

#Normalize the data (images again...)
trainX = trainX / 255.0

for denseLayer in denseLayers:
    for layerSize in layerSizes:
        for conv2DLayer in conv2DLayers:
            modelName = "{}-conv-{}-nodes-{}-dense-{}".format(conv2DLayer, layerSize, denseLayer, int(time.time()))
            tb = TensorBoard(log_dir="logs/{}".format(modelName))
            print("\nNow fitting: ", modelName, "\n")

            model = Sequential()

            model.add(Conv2D(layerSize, kernel_size=(3,3), input_shape = trainX.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for i in range(conv2DLayer-1):
                model.add(Conv2D(layerSize, kernel_size=(3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten(input_shape=(imgSize, imgSize)))

            for i in range(denseLayer):
                model.add(Dense(layerSize))
                model.add(Activation("relu"))

            model.add(Dense(1))
            #Sigmoid function is because it exists between (0 to 1) 
            model.add(Activation("sigmoid"))

            model.compile(loss="binary_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])

            model.fit(trainX, trainY, batch_size=32, epochs=20, validation_split=0.3, callbacks=[tb])