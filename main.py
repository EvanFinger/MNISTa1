import keras
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot

# loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# printing the shapes of the vectors
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))


pyplot.subplot(330 + 1 + 1)
pyplot.imshow(train_X[1], cmap=pyplot.get_cmap('gray'))
pyplot.show()

i0 = np.array(train_X[1])  # 784 inputs
i0 = i0.flatten()
for e in range(784):
    i0[e] = i0[e] / 252
h0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
h1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
o0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
w0 = np.random.uniform(-1, 1, (10, 784))  # first weight layer
w1 = np.random.uniform(-1, 1, (10, 10))  # second weight layer
w2 = np.random.uniform(-1, 1, (10, 10))  # third weight layer
b0 = np.random.uniform(-1, 1, (10, 1))
b1 = np.random.uniform(-1, 1, (10, 1))
b2 = np.random.uniform(-1, 1, (10, 1))


def sig(x):
    return 1/(1 + np.exp(-x))


def softmax(vec):
    exponential = np.exp(vec)
    probabilities = exponential / np.sum(exponential)
    return probabilities


def forward_propagation():
    global o0
    for index in range(10):
        h0[index] = np.sum(np.multiply(i0, w0[index]), 0)  # + b0[index]
        h0[index] = sig(h0[index])
    for index in range(10):
        h1[index] = np.sum(np.multiply(h0, w1[index]), 0)  # + b1[index]
        h1[index] = sig(h1[index])
    for index in range(10):
        o0[index] = np.sum(np.multiply(h1, w2[index]), 0)
    o0 = softmax(o0)



# for image in train_X:
forward_propagation()
print(h0)
print(h1)
print(o0)





