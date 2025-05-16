import numpy as np
from .activations import sigmoid, sigmoid_deriv, relu, relu_deriv

class Layer:
    def __init__(self, input_size, output_size, activation='relu'):
        # Xavier/Glorot initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2./input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(self.input, self.weights) + self.biases
        
        if self.activation == 'sigmoid':
            self.output = sigmoid(self.z)
        elif self.activation == 'relu':
            self.output = relu(self.z)
            
        return self.output

    def backward(self, output_error, learning_rate):
        if self.activation == 'sigmoid':
            delta = output_error * sigmoid_deriv(self.output)
        elif self.activation == 'relu':
            delta = output_error * relu_deriv(self.output)
            
        input_error = np.dot(delta, self.weights.T)
        self.weights -= learning_rate * np.dot(self.input.T, delta)
        self.biases -= learning_rate * np.sum(delta, axis=0, keepdims=True)
        return input_error








# import numpy as np
# from .activations import sigmoid, sigmoid_deriv

# class Layer:
#     def __init__(self, input_size, output_size):
#         self.weights = np.random.randn(input_size, output_size) * 0.01
#         self.biases = np.zeros((1, output_size))

#     def forward(self, input_data):
#         self.input = input_data
#         self.output = sigmoid(np.dot(self.input, self.weights) + self.biases)
#         return self.output

#     def backward(self, output_error, learning_rate):
#         delta = output_error * sigmoid_deriv(self.output)
#         input_error = np.dot(delta, self.weights.T)
#         self.weights -= learning_rate * np.dot(self.input.T, delta)
#         self.biases -= learning_rate * np.sum(delta, axis=0, keepdims=True)
#         return input_error
