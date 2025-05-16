import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)







# import numpy as np

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sigmoid_deriv(x):
#     return x * (1 - x)