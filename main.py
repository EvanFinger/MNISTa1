import os
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

# formatting data
(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape([60000, 784])
temp = labels
labels = np.zeros((60000, 10), dtype=int)
for index in range(60000):
    for index_1 in range(10):
        if index_1 == temp[index]:
            labels[index][index_1] = 1
images = images / 252

wt_in_hid = np.random.uniform(-0.5, 0.5, (20, 784))
wt_hid_out = np.random.uniform(-0.5, 0.5, (10, 20))
bias_in_hid = np.zeros((20, 1))
bias_hid_out = np.zeros((10, 1))

learn_rate = 0.001
nr_correct = 0
epochs = 1000
for epoch in range(epochs):
    for img, l in zip(images, labels):
        img.shape += (1,)
        l.shape += (1,)
        # forward prop in->hid
        z = bias_in_hid + wt_in_hid @ img  # un-activated hidden layer
        a = 1 / (1 + np.exp(-z))  # activated hidden layer
        # forward prop hid->out
        o_pre = bias_hid_out + wt_hid_out @ a  # un-activated output layer
        o = 1 / (1 + np.exp(-o_pre))  # activated output layer
        # cost / error
        e = 1 / len(o) * np.sum((o - l) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(l))

        # back propagation out -> hid (cost funct deriv)
        delta_o = o - l
        wt_hid_out += -learn_rate * delta_o @ np.transpose(a)
        bias_hid_out += -learn_rate * delta_o
        # back propagation hid -> in (activation funct deriv)
        delta_a = np.transpose(wt_hid_out) @ delta_o * (a * (1 - a))
        wt_in_hid += -learn_rate * delta_a @ np.transpose(img)
        bias_in_hid += -learn_rate * delta_a

    print(f"Acc: {round((nr_correct / images.shape[0]) * 100, 2)}%")
    nr_correct = 0

# show results
while True:
    index = int(input("Enter a number (0 - 59999): "))
    img = images[index]
    plt.imshow(img.reshape(28,28), cmap="Greys")

    img.shape += (1,)
    # Forward propagation input -> hidden
    z = bias_in_hid + wt_in_hid @ img.reshape(784, 1)
    a = 1 / (1 + np.exp(-z))
    # Forward propagation hidden -> output
    o_pre = bias_hid_out + wt_hid_out @ a
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(o.argmax())
    plt.show()
