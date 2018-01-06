from __future__ import division

import numpy as np
import struct

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def label_to_vector(label):
    out = np.zeros((10,))
    out[label] = 1.0
    return out

def image_to_vector(image):
    return np.reshape(image, (image.size,)).astype('float64')

def load_pair_mnist(datafile, labelfile):
    return [image_to_vector(i)/255 for i in read_idx(datafile)], [label_to_vector(l) for l in read_idx(labelfile)]

def load_mnist():
    train = load_pair_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte")
    test = load_pair_mnist("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte")
    return train, test

class QuadraticCost(object):
    def __init__(self, nn):
        self.nn = nn

    def value(self, y):
        diff = self.nn.layers[-1].activation - y
        return 1/2*diff.dot(diff)

    def derivative(self, y):
        return self.nn.layers[-1].activation - y

    def delta(y):
        return (self.nn.layers[-1].activation - y)*self.nn.layers[-1].derivative()

class CrossEntropyCost(object):
    def __init__(self, nn):
        self.nn = nn

    def value(self, y):
        a = self.nn.layers[-1].activation
        return -(y.dot(np.log(a)) + (1-y).dot(np.log(1-a)))

    def delta(self, y):
        return (self.nn.layers[-1].activation - y)

class Layer(object):
    """A layer is a collection of neurons"""

    def __init__(self, input_size, output_size):
        """A layer as an input size, and an outptut size that represents the number of neurons in the layer"""
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.zeros((output_size, input_size))
        self.biases = np.zeros((output_size,))
        self.activation = np.zeros((output_size,))

    def forward(self, input_data):
        self.activation = sigmoid(self.biases+self.weights.dot(input_data))
        return self.activation

    def derivative(self):
        return self.activation*(1-self.activation)

    def backward_propagation(self, error, next_layer_weights):
        return (next_layer_weights.T.dot(error))*self.derivative()

    def gradients(self, error, prev_activation):
        self.bias_gradient = error
        self.weights_gradient = np.outer(error, prev_activation)

    def randomize(self):
        self.weights = np.random.normal(loc=0.0, scale=1/np.sqrt(self.input_size), size=self.weights.shape)
        self.biases = np.random.standard_normal(self.biases.shape)

class NeuralNetwork(object):
    """A neural network is a sequence of layers"""

    def __init__(self, sizes, cost='quadratic'):
        """The parameter is a sequence of layer sizes (number of neurons). Note that the first element is the size of your input data and the last one the size of your output"""
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
        if cost == 'quadratic':
            self.cost = QuadraticCost(self)
        elif cost == 'ce':
            self.cost = CrossEntropyCost(self)
        else:
            raise ValueError("Cost must quadratic or cross-entropy (ce)")

    def forward_propagation(self, input_data):
        self.layers[0].forward(input_data)
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i-1].activation)
        return self.layers[-1].activation

    def backward_propagation(self, input_data, input_label):
        delta = self.cost.delta(input_label)
        self.layers[-1].gradients(delta, self.layers[-2].activation)
        for i in range(2, len(self.layers)):
            delta = self.layers[-i].backward_propagation(delta, self.layers[-i+1].weights)
            self.layers[-i].gradients(delta, self.layers[-i-1].activation)
        delta = self.layers[0].backward_propagation(delta, self.layers[1].weights)
        self.layers[0].gradients(delta, input_data)

    def train_batch(self, batch, learning_rate, reg=0.0):
        sum_bias_gradients = [np.zeros_like(l.biases) for l in self.layers]
        sum_weight_gradients = [np.zeros_like(l.weights) for l in self.layers]
        for x, y in batch:
            self.forward_propagation(x)
            self.backward_propagation(x, y)
            for l, sb, sw in zip(self.layers, sum_bias_gradients, sum_weight_gradients):
                sb += l.bias_gradient
                sw += l.weights_gradient

        for l, sb, sw in zip(self.layers, sum_bias_gradients, sum_weight_gradients):
            l.biases -= learning_rate/len(batch)*sb
            l.weights -= learning_rate*(reg*l.weights + sw/len(batch))

    def accuracy(self, test):
        acc = 0
        for sample, label in zip(*test):
            acc += int((self.predict(sample) == label).all())
        acc /= len(test[0])
        return acc

    def GD(self, learning_rate, train, epochs, test, reg=0.0):
        for i in range(epochs):
            self.train_batch(zip(*train), learning_rate, reg=reg/len(train[0]))
            accuracy = self.accuracy(test)
            print("Epoch {} complete : {}".format(i+1, accuracy))

    def SGD(self, learning_rate, train, epochs, batch_size, test, reg=0.0):
        for i in range(epochs):
            zipped_train = zip(*train)
            np.random.shuffle(zipped_train)
            batches = [zipped_train[k:k+batch_size]
                for k in range(0, len(zipped_train), batch_size)]
            for batch in batches:
                self.train_batch(batch, learning_rate, reg=reg/len(zipped_train))
            accuracy = self.accuracy(test)
            print("Epoch {} complete : {}".format(i+1, accuracy))
           
    def predict(self, sample):
        nn_output = self.forward_propagation(sample)
        out = np.zeros_like(nn_output)
        out[np.argmax(nn_output)] = 1.0
        return out

    def randomize(self):
        for layer in self.layers:
            layer.randomize()

if __name__ == '__main__':
    nn = NeuralNetwork([784, 30, 10], cost='ce')
    train, test = load_mnist()
    nn.randomize()
    nn.SGD(0.5, train, 50, 10, test, reg=5.0)
