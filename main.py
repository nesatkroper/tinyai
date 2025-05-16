from neural_network.layer import Layer
from neural_network.network import NeuralNetwork
from data.iris_loader import load_data
import numpy as np
import pickle

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Split train into train/val
    val_size = int(0.2 * len(X_train))
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_train, y_train = X_train[val_size:], y_train[val_size:]

    nn = NeuralNetwork()
    nn.add(Layer(4, 16, activation='relu'))  
    nn.add(Layer(16, 3, activation='sigmoid'))  

    print("Training...")
    nn.train(X_train, y_train, X_val, y_val, epochs=200, lr=0.01, batch_size=16)

    # Evaluate
    test_preds = nn.predict(X_test)
    test_acc = np.mean(np.argmax(test_preds, axis=1) == np.argmax(y_test, axis=1))
    print(f"Test Accuracy: {test_acc:.4f}")
    with open('iris_model.pkl', 'wb') as f:
        pickle.dump(nn, f)

if __name__ == "__main__":
    main()





