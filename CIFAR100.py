import numpy as np
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

#preprocessing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow(x_train,y_train)

test_set = test_datagen.flow(x_test,y_test)

from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,LeakyReLU,Dropout
from keras.models import Sequential

classifier=Sequential()

classifier.add(Conv2D(128,(3,3),activation='relu',input_shape=(32,32,3)))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(rate=0.2))

classifier.add(Conv2D(64,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(rate=0.2))

classifier.add(Conv2D(128,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(rate=0.2))

classifier.add(Flatten())

classifier.add(Dense(128,activation='relu'))

classifier.add(Dense(256,activation='relu'))

classifier.add(Dense(100,activation='softmax'))

classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

classifier.fit_generator(training_set,steps_per_epoch = 8000,epochs = 100,validation_data = test_set,validation_steps = 2000)
