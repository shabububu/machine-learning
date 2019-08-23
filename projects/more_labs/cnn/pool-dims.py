## From: https://stackoverflow.com/questions/39930952/cannot-import-keras-after-installation
# 
# virtualenv -p python3 py-keras
# source py-keras/bin/activate
# pip install -q -U pip setuptools wheel
# pip3 install tensorflow
# python3 pool-dims.py
# deactivate

from keras import Sequential
from keras.layers import MaxPooling2D

model = Sequential()
model.add(MaxPooling2D(pool_size=2, strides=2, input_shape=(100, 100, 15)))
model.summary()




