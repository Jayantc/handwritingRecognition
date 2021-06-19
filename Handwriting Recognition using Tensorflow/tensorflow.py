import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# mnist dataset parameters
no_of_classes= 10 # classes(0 to 9)
no_of_features= 784 # 28*28 image shape
data= mnist.load_data()
print(data)
(X_train, y_train), (X_test, y_test)= data

# Converting format to float32
X_train, X_test= np.array(X_train, np.float32), np.array(X_test, np.float32)

# flatten imaged to 1-D vector of 784 features
X_train, X_test= X_train.reshape([-1, no_of_features]), X_test.reshape([-1, no_of_features])

# Normalize images values from [0, 255] to [0, 1]
X_train, X_test= X_train/255, X_test/255

# Representing input data
def display(num):
    label= y_train[num]
    image= X_train[num].reshape([28, 28])
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
display(2000)

images= X_train[0].reshape([1, 784])
for i in range(1, 500):
    images= np.concatenate((images, X_train[i].reshape([1, 784])))
plt.imshow(images, cmap=plt.get_cmap('gray_r'))
plt.show()


# Training Parameters
learning_rate= 0.001
training_steps= 3000
batch_size= 250
display_steps= 100
n_hidden= 512

# Shuffle and batch data
train_data= tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_data= train_data.repeat().shuffle(60000).batch(batch_size).prefetch(1)

# Store layers weight & bias
random_normal= tf.initializers.RandomNormal()

weights= {
    'h': tf.Variable(random_normal([no_of_features, n_hidden])),
    'out': tf.Variable(random_normal([n_hidden, no_of_classes]))
}
bias= {
    'b': tf.Variable(tf.zeros([n_hidden])),
    'out': tf.Variable(tf.zeros([no_of_classes]))
}

# Create Model
def create_model(input):
    # Hidden fully connected layer with 512 neurons
    hidden_layers= tf.add(tf.matmul(input, weights['h']), bias['b'])
    # Apply sigmoid to hidden layers output for non-linearity
    hidden_layers= tf.nn.sigmoid(hidden_layers)
    # Output fully connected layer with a neuron for each class
    output_layer= tf.matmul(hidden_layers, weights['out'])+bias['out']
    # Apply softmax to normalize the logits to a probability distribution
    return tf.nn.softmax(output_layer)

# Loss Function
def cross_entropy(y_pred, y_true):
    # Encode lebel to a one hot vector
    y_true= tf.one_hot(y_true, depth=no_of_classes)
    # Clip prediction values to avoid log(0) error
    y_pred= tf.clip_by_value(y_pred, 1e-9, 1.)
    # compute cross entropy
    return tf.reduce_mean(-tf.reduce_sum(y_true*tf.math.log(y_pred)))

# Set optimizer
optimizer= tf.keras.optimizers.SGD(learning_rate)
def run_optimizer(x, y):
    # wrap computation inside a gradientape for automatic differentiation
    with tf.GradientTape() as g:
        pred= create_model(x)
        loss= cross_entropy(pred, y)
    # variables to update i.e. trainable variables
    trainable_variables= list(weights.values()) + list(bias.values())
    # Compute gradients
    gradients= g.gradient(loss, trainable_variables)
    # Update w, b following gradients
    optimizer.apply_gradients(zip(gradients, trainable_variables))

# Create accuracy metrics
def accuracy(y_pred, y_true):
    correct_prediction= tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# Run training for given no. of steps
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # run optimization to update w, b values
    run_optimizer(batch_x, batch_y)
    if step%display_steps== 0:
        pred= create_model(batch_x)
        loss= cross_entropy(pred, batch_y)
        acc= accuracy(pred, batch_y)
        print("Training epochs : ", step, " loss : ", loss, " accuracy : ", acc)

# Test model on validation set
pred= create_model(X_test)
print('Test accuracy : ', accuracy(pred, y_test))

n_images= 200
test_image= X_test[:n_images]
test_lebels= y_test[:n_images]
prediction= create_model(test_image)
for i in range(n_images):
    model_prediction= np.argmax(prediction.numpy()[i])
    if(model_prediction != test_lebels[i]):
        plt.imshow(np.reshape(test_image[i], [28, 28]), cmap='gray_r')
        plt.show()
        print('Original lebels : ', test_lebels[i])
        print('Model prediction : ', model_prediction)