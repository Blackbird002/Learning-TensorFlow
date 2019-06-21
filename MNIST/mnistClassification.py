#!/usr/bin/python3

'''
MNIST: Modified National Institute of Standards and Technology database
    -It is a large database of handwritten digits that is commonly used 
     for training various image processing systems
    -MNIST database contains 60,000 training images and 10,000 testing 
     images taken from American Census Bureau employees and American high 
'''

import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as mplt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam

trainImages = []
trainLabels = []
testImages = []
testLabels = []

#MNIST directly from TensorFlow, import and load the data:
minst = tf.keras.datasets.mnist
(trainImages, trainLabels), (testImages, testLabels) = minst.load_data()

def showImage(index, data):
    mplt.imshow(data[index], cmap='Greys')
    mplt.show()

def show64ImagesAndLabels(images, labels):
    mplt.figure(figsize=(10,10))
    for i in range(64):
        mplt.subplot(8,8,i+1)
        mplt.xticks([])
        mplt.yticks([])
        mplt.grid(False)
        mplt.imshow(trainImages[i],cmap='Greys')
        mplt.xlabel(trainLabels[i])
    mplt.show()

def showImageAndLabel(images,labels,i):
    mplt.imshow(trainImages[i],cmap='Greys')
    mplt.xlabel(trainLabels[i])
    mplt.show()

def printArrayInfo(trainImages,trainLabels):
    print("Images Array Shape: ", trainImages.shape)
    print("Images Length: ", len(trainImages))
    print("Labels length: ", len(trainLabels))

printArrayInfo(trainImages,trainLabels)
#showImageAndLabel(trainImages,trainLabels,144)

#Reshaping the array to 4-dims so that it can work with the Keras API
trainImages = trainImages.reshape(trainImages.shape[0],28,28,1)
testImages = testImages.reshape(testImages.shape[0],28,28,1)

print("After reshape: ")
printArrayInfo(trainImages,trainLabels)

# Making sure that the values are float
trainImages = trainImages.astype('float32')
testImages = testImages.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
trainImages /= 255
testImages /= 255

#Categorize the labels (not required here...)
#trainLabels = np_utils.to_categorical(trainLabels,10)
#testLabels = np_utils.to_categorical(testLabels,10)

#Building the Convolutional Neural Network (CNN)
model = Sequential([
    #transforms the format of the images from a 2d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels
    Flatten(),
    #Densely-connected, or fully-connected, neural layers. 
    Dense(128, activation=tf.nn.relu),
    Dense(128, activation=tf.nn.relu),
    #The last layer is a 10-node softmax layer—this returns an array of 10 probability scores that sum to 1
    Dense(10, activation=tf.nn.softmax)
])

'''
Loss function —This measures how accurate the model is during training. 
            We want to minimize this function to "steer" the model in the right direction.
Optimizer —This is how the model is updated based on the data it sees and its loss function.
Metrics —Used to monitor the training and testing steps. The following uses accuracy, 
         the fraction of the images that are correctly classified.
'''
#Compile Model
model.compile(optimizer=Adam(), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

#Fit the model
model.fit(trainImages,trainLabels, epochs=10)

#Evaluate model
model.evaluate(testImages,testLabels)

def predict(index):
    title = "Prediction at index: " + str(index)
    mplt.figure(num=title)
    image_index = index
    mplt.imshow(testImages[image_index].reshape(28, 28),cmap='Greys')
    pred = model.predict(testImages[image_index].reshape(1,28,28,1))
    mplt.xlabel(pred.argmax())
    mplt.show()

#Predict the first 64 images in the test dataset
def predict64Images(images):
    mplt.figure(num="Predictions using MNIST test DB",figsize=(10,10))
    for i in range(64):
        mplt.subplot(8,8,i+1)
        mplt.xticks([])
        mplt.yticks([])
        mplt.grid(False)

        mplt.imshow(images[i].reshape(28, 28),cmap='Greys')
        pred = model.predict(images[i].reshape(1,28,28,1))
        mplt.xlabel(pred.argmax())
    mplt.show()

predict(144)
predict64Images(testImages)