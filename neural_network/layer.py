import numpy as np
from .activations import sigmoid, sigmoid_deriv

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        self.output = sigmoid(np.dot(self.input, self.weights) + self.biases)
        return self.output

    def backward(self, output_error, learning_rate):
        delta = output_error * sigmoid_deriv(self.output)
        input_error = np.dot(delta, self.weights.T)
        self.weights -= learning_rate * np.dot(self.input.T, delta)
        self.biases -= learning_rate * np.sum(delta, axis=0, keepdims=True)
        return input_error
