from neural_network.layer import Layer
from neural_network.network import NeuralNetwork
from data.iris_loader import load_data
import numpy as np

def main():
    X_train, X_test, y_train, y_test = load_data()

    nn = NeuralNetwork()
    nn.add(Layer(4, 8))  # Input: 4 features → 8 neurons
    nn.add(Layer(8, 3))  # Hidden: 8 → Output: 3 classes

    print("Training...")
    nn.train(X_train, y_train, epochs=200, lr=0.1)

    # Evaluate
    correct = 0
    for i in range(len(X_test)):
        pred = nn.predict(X_test[i:i+1])
        if np.argmax(pred) == np.argmax(y_test[i]):
            correct += 1

    print("Accuracy:", correct / len(X_test))

if __name__ == "__main__":
    main()

