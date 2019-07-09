import cv2
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
    
categories = ["Dog", "Cat"]
imgSize = 128

root = tk.Tk()
root.withdraw()

def showIns():
    messagebox.showinfo("Cat and Dog recognizer", "Please select an image of a cat or dog")

def getFilePath():
    filePath = filedialog.askopenfilename()
    return filePath

def setUpImage(imgPath):
    imgArr = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    newArr = cv2.resize(imgArr, (imgSize, imgSize))
    return newArr.reshape(-1, imgSize, imgSize, 1)

def getModel(name):
    return tf.keras.models.load_model(name)

def predict(model, imgArr):
    prediction = model.predict(imgArr)
    predVal = int(prediction[0][0])
    print(categories[predVal])
    messagebox.showinfo("Prediction", "It is a " + categories[predVal])

def main():
    showIns()
    imgPath = getFilePath()
    imgArr = setUpImage(imgPath)
    model = getModel("CatsAndDogs.model")
    predict(model, imgArr)

if __name__ == "__main__":
    main()