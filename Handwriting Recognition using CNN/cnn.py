import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as k

(image_train, lebel_train), (image_test, lebel_test)= mnist.load_data()

if k.image_data_format()=='channels_first':
    X_train= image_train.reshape(image_train.shape[0], 1, 28, 28)
    X_test = image_test.reshape(image_test.shape[0], 1, 28, 28)
    input_shape=(1, 28, 28)
else:
    X_train = image_train.reshape(image_train.shape[0], 28, 28, 1)
    X_test = image_test.reshape(image_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
X_train= X_train.astype('float32')
X_test= X_test.astype('float32')
X_train= X_train/255
X_test= X_test/255

y_train= tensorflow.keras.utils.to_categorical(lebel_train, 10)
y_test= tensorflow.keras.utils.to_categorical(lebel_test, 10)

def display_sample(num):
    # one-hot array of sample's lebel
    print(y_train[[num]])
    # lebel converted back to number
    lebel= lebel_train[num].argmax(axis=0)
    # Reshape 784 to 28*28
    image= X_train[num].reshape([28, 28])
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
display_sample(1000)

model= Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape= input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history= model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=2, validation_data=(X_test, y_test))

score= model.evaluate(X_test, y_test, verbose=0)
print('Test loss : ', score[0])
print('Test accuracy : ', score[1])