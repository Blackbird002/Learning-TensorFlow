import numpy as np
import matplotlib.pyplot as plt
import os
#tool for fast prototyping of computer vision problems
import cv2
import random
#binary protocols for serializing and de-serializing a Python object
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

'''
Location of Dog & Cat Images from Microsoft's Kaggle Cats and Dogs
dataset:
https://www.microsoft.com/en-us/download/details.aspx?id=54765
'''
dataDir = "/home/rayshash/PetImages"

'''
0 - dog
1 - cat
'''
categories = ["Dog", "Cat"]
imgSize = 128

#Optional (testing)
def loadImages():
    for category in categories:
        #Joins path components intelligently
        path = os.path.join(dataDir, category)
        #For every image in the directory...
        for img in os.listdir(path):
            imgArr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            plt.imshow(imgArr, cmap="gray")
            plt.show()
            break
        break
        print(imgArr)
        print(imgArr.shape)
    return imgArr

#Optional (testing)
#Need to normalize picture shapes (pictures are different shape)
def normalizeImgSize(imgArr):
    resizeArr = cv2.resize(imgArr, (imgSize,imgSize))
    plt.imshow(resizeArr)
    plt.show()
    return resizeArr

def createTrainData(trainData):
    for category in categories:
        path = os.path.join(dataDir, category)
        classNum = categories.index(category)
        for img in os.listdir(path):
            #Some images are broken - avoids OS errors and warnings
            try:
                imgArr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                resizeArr = cv2.resize(imgArr, (imgSize,imgSize))
                trainData.append([resizeArr, classNum])
            except Exception as e:
                pass
    print("Items in trainData: ", len(trainData))
    #Shuffle the data so it's not all dogs and then all cats! (better training)
    random.shuffle(trainData)
    return trainData

def reshapeData(trainData, trainX, trainY):
    #Bulid the X and Y list...
    for features, label in trainData:
        trainX.append(features)
        trainY.append(label)

    #trainX has to be a numpy array - and shape has to be specified
    trainX = np.array(trainX).reshape(-1, imgSize, imgSize, 1)
    return (trainX,trainY)

#Save trainX and trainY into files...
def saveData(trainX, trainY): 
    pickleOut = open("X.pickle", "wb")
    pickle.dump(trainX, pickleOut)
    pickleOut.close()

    pickleOut = open("Y.pickle", "wb")
    pickle.dump(trainY, pickleOut)
    pickleOut.close()

#Load saved python serialized objects...
def loadData(trainX, trainY):
    trainX = pickle.load(open("X.pickle", "rb"))
    trainY = pickle.load(open("Y.pickle", "rb"))
    return (trainX, trainY)

#Building the model (CNN)
'''
 -ReLU (Rectified Linear Unit) Activation Function
 -Sigmoid or Logistic Activation Function
'''
def bulidCNNModel(trainX):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3,3), input_shape = trainX.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, kernel_size=(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten(input_shape=(imgSize, imgSize)))

    model.add(Dense(128))
    model.add(Activation("relu"))

    model.add(Dense(1))
    #Sigmoid function is because it exists between (0 to 1) 
    model.add(Activation("sigmoid"))

    return model

#Compile the model
def compileModel(model):
    model.compile(loss="binary_crossentropy",
                optimizer="adam",
                metrics=["accuracy"])
    return model

#Train...
def fitModel(model, trainX, trainY):    
    model.fit(trainX, trainY, batch_size=32, epochs=7, validation_split=0.1)
    return model

#Save the model
def saveModel(model, modelName): 
    print("Saving the model...")
    model.save("{}.model".format(modelName))


def main():
    #trainData = []
    #trainData = createTrainData(trainData)
    trainX = []
    trainY = []
    #(trainX, trainY) = reshapeData(trainData, trainX, trainY)
    #saveData(trainX, trainY)
    (trainX, trainY) = loadData(trainX, trainY)

    #Normalize the data (images again...)
    trainX = trainX / 255.0

    model = bulidCNNModel(trainX)
    model = compileModel(model)
    model = fitModel(model, trainX, trainY)

    saveModel(model, "CatsAndDogs")

if __name__ == "__main__":
    main()