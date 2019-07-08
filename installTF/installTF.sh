#!/bin/bash

#Source: https://www.tensorflow.org/install/pip#system-requirements
#Virtualenv: https://virtualenv.pypa.io/en/stable/


function installCurl(){
    #Check if curl is installed...
    if ! dpkg-query -s curl 1> /dev/null 2>&1 ; then
        echo "Package curl is not currently installed..."
        echo "Installing curl..."
        sleep 2
        sudo apt install curl
    else
        echo "Package curl is currently installed!"
    fi
  return
}

function installPip(){
    #Check if pip is installed...
    if ! dpkg-query -s curl 1> /dev/null 2>&1 ; then
        echo "Package pip is not currently installed..."
        echo "Installing pip..."
        sleep 2
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python get-pip.py
    else
        echo "Package pip is currently installed!"
    fi
    return
}

function installPython3Dev(){
    #Check if python3-dev is installed...
    if ! dpkg-query -s python3-dev 1> /dev/null 2>&1 ; then
        echo "Package python3-dev is not currently installed..."
        echo "Installing python3-dev..."
        sleep 2
        sudo apt install python3-dev
    else
        echo "Package python3-dev is currently installed!"
    fi
    sudo pip3 install -U virtualenv  # system-wide install
    return
}

function createVirEnv(){
    #Create a new virtual environment by choosing a Python interpreter and making a ./venv directory to hold it:
    #Python virtual environments are used to isolate package installation from the system
    virtualenv --system-site-packages -p python3 ./venv
    return
}

function activateVirEnv(){
    #When virtualenv is active, shell prompt is prefixed with (venv).
    source ./venv/bin/activate  # sh, bash, ksh, or zsh
    return
}

function installTensorFlowPip(){
    #Install packages within a virtual environment without affecting the host system setup...
    echo "Installing TensorFlow pip package..."
    sleep 2
    pip install --upgrade pip
    pip list  # show packages installed within the virtual environment
    sleep 2
    if [["$0" = "gpu"]]; then
        echo "Installing gpu tensorflow"
        pip install --upgrade tensorflow-gpu
    else
        echo "Installing cpu tensorflow"
        pip install --upgrade tensorflow
    fi
    python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
    sleep 2
    return
}

function exitVirEnv(){
    deactivate  # don't exit until you're done using TensorFlow
    return
}

function run(){
    installCurl
    installPip
    sudo apt update
    installPython3Dev
    createVirEnv
    activateVirEnv
    installTensorFlowPip
    echo "Done! Exiting virtual enviroment..."
    sleep 5
    exitVirEnv
}

#Runs the script...
run