# Learning-TensorFlow
My self-study notes and scripts for learning TensorFlow

## Helpful videos:
"sentdex" is a great youtuber that explains everything very well:
- [Deep Learning basics with Python, TensorFlow and Keras Playlist - sentdex](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN)
- [Machine Learning with Python - sentdex](https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v)

Official TensorFlow Youtube channel:
- [Coding TensorFlow](https://www.youtube.com/playlist?list=PLQY2H8rRoyvwLbzbnKJ59NkZvQAW9wLbx)

## Installing TensorFlow:

### CPU (Ubuntu 18 64-bit):
- [Install TensorFlow](https://www.tensorflow.org/install)
- [Easy Bash Script to install TF - (Linux pip & virtual env) ](https://github.com/Blackbird002/Learning-TensorFlow/blob/master/installTF.sh)

### GPU (Ubuntu 18.10 64-bit):
- Need an Nvidia GPU with CUDA Compute >= 3.5

We need to blacklist the nouveau dirver to prevent modprobe from loading it via modalias:
```
sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"

sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
```

Make sure it worked:
```
cat /etc/modprobe.d/blacklist-nvidia-nouveau.conf
```
It should say: 
```
blacklist nouveau
options nouveau modeset=0
```

Update kernel initramfs and then reboot:
```
sudo update-initramfs -u
sudo reboot
```

Add the graphics driver PPA and update:
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
```

If you already have an older verison of Nvidia GPU drivers installed:
```
sudo apt-get purge nvidia*
sudo reboot
```

Install latest proprietary Nvidia driver (latest verison when I last installed 430.86):
```
sudo apt update
sudo ubuntu-drivers autoinstall
sudo reboot
```

Verify that nvidia driver was successfully installed:
```
nvidia-smi
```

Install Anaconda (Popular Python Data Science Platform):
[Anaconda 2019.03 for Linux Installer Python 3.7 version](https://www.anaconda.com/distribution/#download-section)

Create a new virtual environment with python 3.7:
```
conda create -n tf-gpu python=3.7
source activate tf-gpu
```
Remember to activate the virtual environment (called tf-gpu in this case) everytime you need to run TensorFlow!

Install Tensorflow GPU along with CUDA & CuDNN:
```
conda install tensorflow-gpu
conda install cudatoolkit
conda install cudnn
conda install h5py
```

If you need to install other python packages like Keras, you can use conda seach:
```
conda search Keras
```

## MNIST Classification
- Modified National Institute of Standards and Technology database
- Database of handwritten digits
- Training set: 60,000 examples & labels
- Test set: 10,000 examples & labels
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [mnistClassification.py](https://github.com/Blackbird002/Learning-TensorFlow/blob/master/MNIST/mnistClassification.py)

## Fashion MNIST Classification
  - Database of grayscale images of 10 different categories of clothes
  - [FashionMnistClassification.py](https://github.com/Blackbird002/Learning-TensorFlow/blob/master/Fashion%20MNIST/FashionMnistClassification.py)

## Classifying Cats and Dogs (using TensorBoard for optimization)
  - Dataset: [Microsoft's Kaggle Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
  - [CatsAndDogs.py](https://github.com/Blackbird002/Learning-TensorFlow/blob/master/CatsAndDogs/CatsAndDogs.py)

  Optimization - trying to find the best model for this example
  - [tryModels.py](https://github.com/Blackbird002/Learning-TensorFlow/blob/master/CatsAndDogs/tryModels.py)
  - [Model metrics](https://github.com/Blackbird002/Learning-TensorFlow/tree/master/CatsAndDogs)
