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


def set_inout(x):
    global i0
    global trueout
    i0 = np.array(train_X[x])  # 784 inputs
    i0 = i0.flatten()
    for e in range(784):
        i0[e] = i0[e] / 252
    trueout = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for element in range(10):
        if element == train_y[x]:
            trueout[train_y[x]] = 1


z0 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
s0 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
w0 = np.random.uniform(-1, 1, (10, 784))  # first weight layer
w1 = np.random.uniform(-1, 1, (10, 10))  # second weight layer

b0 = np.random.uniform(-1, 1, (1, 10))
b1 = np.random.uniform(-1, 1, (1, 10))



def sig(x):
    return 1/(1 + np.exp(-x))


def softmax(vec):
    exponential = np.exp(vec)
    probabilities = exponential / np.sum(exponential)
    return probabilities


def forward_propagation():
    global s0
    global i0
    global s0_a
    for index in range(10):
        z0[index] = np.sum(np.multiply(i0, w0[index]), 0) + b0[0][index]
        z0[index] = sig(z0[index])
    for index in range(10):
        s0[index] = np.sum(np.multiply(z0, w1[index]), 0) + b1[0][index]
    s0_a = softmax(s0)


def predict():
    temp = 0
    prediction = 0
    for index in range(10):
        if s0[index] > temp:
            temp = s0[index]
            prediction = index
    return prediction


def cost():
    global trueout
    return np.square(np.subtract(trueout, s0))


def back_propagation():
    for w1_r in range(10):
        for w1_c in range(10):
            w1[w1_r][w1_c] = w1[w1_r][w1_c] - (2 * (s0_a[w1_r] - trueout[w1_r]))*(s0[w1_r])*(z0[w1_c])


    print(s0_a)
    print(trueout)
    print(cost())
    print(cost().mean())




# for image in train_X:
for index in range(1):
    set_inout(index)
    forward_propagation()
    back_propagation()
    print(predict())





