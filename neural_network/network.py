class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, x_train, y_train, epochs, lr):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(x_train)):
                output = x_train[i:i+1]
                for layer in self.layers:
                    output = layer.forward(output)

                error = y_train[i:i+1] - output
                total_error += (error ** 2).mean()

                grad = 2 * error
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, lr)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_error / len(x_train):.4f}")



