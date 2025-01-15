from tensorflow import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

np.savetxt('x_train.txt', train_images.reshape(-1, 28*28), delimiter=',')
np.savetxt('y_train.txt', train_labels, delimiter=',')

np.savetxt('x_test.txt', test_images.reshape(-1, 28*28), delimiter=',')
np.savetxt('y_test.txt', test_labels, delimiter=',')