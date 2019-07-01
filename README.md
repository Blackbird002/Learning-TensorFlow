# Learning-TensorFlow
My self-study notes and scripts for learning TensorFlow

## Installing TensorFlow

### CPU:
- [Install TensorFlow](https://www.tensorflow.org/install)
- [Easy Bash Script to install TF - (Linux pip & virtual env) ](https://github.com/Blackbird002/Learning-TensorFlow/blob/master/installTF.sh)

### GPU:
- Need an Nvidia GPU with CUDA Compute >= 3.5
- Ubuntu 18.10 64-bit

We need to blacklist the nouveau dirver to prevent modprobe from loading it via modalias:
```
sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"

sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
```

Make sure it worked:
```
cat /etc/modprobe.d/blacklist-nvidia-nouveau.conf
```

Update kernel initramfs and then reboot:
```
sudo update-initramfs -u
sudo reboot
```

## MNIST (The "Hello World" of ML)
- Modified National Institute of Standards and Technology database
- Database of handwritten digits
- Training set: 60,000 examples & labels
- Test set: 10,000 examples & labels
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Script](https://github.com/Blackbird002/Learning-TensorFlow/blob/master/MNIST/mnistClassification.py)

## Fashion MNIST
  - Database of grayscale images of 10 different categories of clothes
  - [TensorFlow Script](https://github.com/Blackbird002/Learning-TensorFlow/blob/master/Fashion%20MNIST/FashionMnistClassification.py)