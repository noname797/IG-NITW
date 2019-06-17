# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 11:34:26 2019

@author: noname
"""

import tensorflow as tf
import numpy as np
Mnist= tf.keras.datasets.mnist
(Train_imgs, train_labels),(test_imgs, test_labels)=Mnist.load_data ()
Train_imgs=Train_imgs.astype('float32')
test_imgs=test_imgs.astype('float32')
Train_imgs/=255
test_imgs/=255

import keras
from keras.models import Sequential
from keras.layers import Flatten,Dense

classifier=Sequential()

classifier.add(Flatten(input_shape=(28,28)))

classifier.add(Dense(128,activation='relu'))

classifier.add(Dense(10,activation='softmax'))

classifier.compile(optimizer='adam',metrics=['accuracy'],loss='sparse_categorical_crossentropy')

classifier.fit(Train_imgs,train_labels,epochs=30,validation_split=0.1)

y_pred=classifier.predict(test_imgs) #predicted multi output with probabilities

y_pred_num=np.argmax(y_pred) #predicted single output

evaluation=classifier.evaluate(test_imgs,test_labels) #gives loss value and metrics
