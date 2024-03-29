from __future__ import absolute_import, division, print_function
'''
Fashion MNIST: Modified National Institute of Standards and Technology database
    -It is a large database of grayscale images of 10 different categories of clothes
    -Fashion MNIST database contains 60,000 training images and 10,000 testing 
     images 

    Labels: 
    0	T-shirt/top
    1	Trouser
    2	Pullover
    3	Dress
    4	Coat
    5	Sandal
    6	Shirt
    7	Sneaker
    8	Bag
    9	Ankle boot
'''

import numpy as np 
import tensorflow as tf
import pickle
import matplotlib.pyplot as mplt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical, normalize

trainImages = []
trainLabels = []
testImages = []
testLabels = []

#MNIST directly from TensorFlow, import and load the data:
fminst = tf.keras.datasets.fashion_mnist
(trainImages, trainLabels), (testImages, testLabels) = fminst.load_data()

#Seeing the data type of downloaded data
print("trainImage is a:",type(trainImages))
print("trainLabels is a:",type(trainLabels))

''' 
Normalizing the RGB codes by dividing it to the max RGB value.
trainImages /= 255
testImages /= 255
'''
trainImages = normalize(trainImages)
testImages = normalize(testImages)

#Reshaping the array to 4-dims so that it can work with the Keras API 
trainImages = trainImages.reshape(trainImages.shape[0],28,28,1)
testImages = testImages.reshape(testImages.shape[0],28,28,1)


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

def printType(label):
    switcher = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }
    return switcher.get(label, "N/A")

def bulidCNNModel(trainX):
    model = Sequential()

    model.add(Conv2D(28,kernel_size=(4,4),input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten(input_shape=(28,28)))

    model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.1))

    model.add(Dense(10, activation='softmax'))
    return model

'''
Loss function —This measures how accurate the model is during training. 
        We want to minimize this function to "steer" the model in the right direction.
Optimizer —This is how the model is updated based on the data it sees and its loss function.
Metrics —Used to monitor the training and testing steps. The following uses accuracy, 
         the fraction of the images that are correctly classified.
'''
#Compile Model
def compileModel(model):
    model.compile(optimizer=Adam(), 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    return model

#Fit the model
def fitModel(model, trainX, trainY):
    model.fit(trainX, trainY, epochs=16)
    return model

#Evaluate model
def evalModel(model, testX, testY):
    print("Evaluating the model...")
    model.evaluate(testX, testY)

#Save the model
def saveModel(model, modelName): 
    print("Saving the model...")
    model.save("{}.model".format(modelName))

def predict(index, model):
    title = "Prediction at index: " + str(index)
    mplt.figure(num=title)
    image_index = index
    mplt.imshow(testImages[image_index].reshape(28, 28),cmap='Greys')
    pred = model.predict(testImages[image_index].reshape(1,28,28,1))
    mplt.xlabel(printType(pred.argmax()))
    mplt.show()

#Predict the first 64 images in the test dataset
def predict64Images(images, model):
    mplt.figure(num="Predictions using MNIST test DB",figsize=(10,10))
    for i in range(64):
        mplt.subplot(8,8,i+1)
        mplt.xticks([])
        mplt.yticks([])
        mplt.grid(False)

        mplt.imshow(images[i].reshape(28, 28),cmap='Greys')
        pred = model.predict(images[i].reshape(1,28,28,1))
        mplt.xlabel(printType(pred.argmax()))
    mplt.show()

#Save trainX and trainY into files...
def saveData(trainX, trainY): 
    pickleOut = open("X.pickle", "wb")
    pickle.dump(trainX, pickleOut)
    pickleOut.close()

    pickleOut = open("Y.pickle", "wb")
    pickle.dump(trainY, pickleOut)
    pickleOut.close()

def main():
    printArrayInfo(trainImages,trainLabels)
    #showImageAndLabel(trainImages,trainLabels,144)

    model = bulidCNNModel(trainImages)
    model = compileModel(model)
    model = fitModel(model, trainImages, trainLabels)

    predict(144, model)
    predict64Images(testImages, model)
    evalModel(model, testImages, testLabels)
    saveModel(model, "Fashion")

if __name__ == "__main__":
    main()