import numpy as np
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
x_train=x_train/255
x_test=x_test/255
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,LeakyReLU,Dropout
from keras.models import Sequential

classifier=Sequential()

classifier.add(Conv2D(128,(3,3),activation='relu',input_shape=(32,32,3)))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(rate=0.2))

classifier.add(Conv2D(128,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(rate=0.2))

classifier.add(Conv2D(128,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(rate=0.2))

classifier.add(Flatten())

classifier.add(Dense(512,activation='relu'))
classifier.add(Dropout(0.2)
classifier.add(Dense(512,activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(100,activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9)
classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

classifier.fit_generator(training_set,epochs = 100,batch_size=64)
