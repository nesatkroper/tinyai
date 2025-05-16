import numpy as np  # Add this at the top

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, x_train, y_train, x_val, y_val, epochs, lr, batch_size=32):
        for epoch in range(epochs):
            # Shuffle data
            permutation = np.random.permutation(len(x_train))
            x_train_shuffled = x_train[permutation]
            y_train_shuffled = y_train[permutation]
            
            total_error = 0
            for i in range(0, len(x_train_shuffled), batch_size):
                # Get batch
                x_batch = x_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # Forward pass
                output = x_batch
                for layer in self.layers:
                    output = layer.forward(output)
                
                # Compute error
                error = y_batch - output
                total_error += np.mean(error**2)
                
                # Backward pass
                grad = -2 * error / batch_size
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, lr)
            
            # Validation
            if epoch % 10 == 0:
                val_preds = self.predict(x_val)
                val_acc = np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val, axis=1))
                print(f"Epoch {epoch}, Loss: {total_error/len(x_train_shuffled):.4f}, Val Acc: {val_acc:.4f}")

        # best_val = 0
        # for epoch in range(epochs):
        #     # ... training code ...
        #     if val_acc > best_val:
        #         best_val = val_acc
        #         # Save best model
        #     elif epoch > 50 and val_acc < best_val - 0.05:
        #         print("Early stopping")
        #         break





# class NeuralNetwork:
#     def __init__(self):
#         self.layers = []

#     def add(self, layer):
#         self.layers.append(layer)

#     def predict(self, x):
#         for layer in self.layers:
#             x = layer.forward(x)
#         return x

#     def train(self, x_train, y_train, epochs, lr):
#         for epoch in range(epochs):
#             total_error = 0
#             for i in range(len(x_train)):
#                 output = x_train[i:i+1]
#                 for layer in self.layers:
#                     output = layer.forward(output)

#                 error = y_train[i:i+1] - output
#                 total_error += (error ** 2).mean()

#                 grad = 2 * error
#                 for layer in reversed(self.layers):
#                     grad = layer.backward(grad, lr)

#             if epoch % 10 == 0:
#                 print(f"Epoch {epoch}, Loss: {total_error / len(x_train):.4f}")



