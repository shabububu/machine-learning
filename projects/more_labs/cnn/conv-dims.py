## From: https://stackoverflow.com/questions/39930952/cannot-import-keras-after-installation
# 
# virtualenv -p python3 py-keras
# source py-keras/bin/activate
# pip install -q -U pip setuptools wheel
# pip3 install tensorflow
# python3 conv-dims.py
# deactivate

from keras import Sequential
from keras.layers.convolutional import Conv2D

model = Sequential()
#model.add(Conv2D(filters=16, kernel_size=2, strides=2, padding='valid', activation='relu', input_shape=(200, 200, 1)))
model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu', input_shape=(128, 128, 3)))
model.summary()

