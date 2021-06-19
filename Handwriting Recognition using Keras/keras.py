from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np

(image_train, lebel_train), (image_test, lebel_test)= mnist.load_data()

X_train= image_train.reshape(60000, 784)
X_test= image_test.reshape(10000, 784)
X_train= X_train.astype('float32')
X_test= X_test.astype('float32')
X_train= X_train/255
X_test= X_test/255

y_train= keras.utils.to_categorical(lebel_train, 10)
y_test= keras.utils.to_categorical(lebel_test, 10)

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

# Creating model
model= Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

print(model.summary())

#Loss Function
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

#Training
history= model.fit(X_train, y_train, batch_size=100, epochs=10, verbose=2, validation_data=(X_test, y_test))

score= model.evaluate(X_test, y_test, verbose=0)
print('Test loss : ', score[0])
print('Test accuracy : ', score[1])

test= np.zeros(shape=(1, 784))
for x in range(1000):
    test= X_test[x, :].reshape(1, 784)
    predicted= model.predict(test).argmax()
    lebel= y_test[x].argmax()
    if(predicted != lebel):
        print('Prediction is ', predicted, ' original is ', lebel)
        plt.imshow(test.reshape([28, 28]), cmap=plt.get_cmap('gray_r'))
        plt.show()