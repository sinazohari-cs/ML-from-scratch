import numpy as np

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight = 0
        self.bias = 0

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.epochs):
            y_pred = self.weight * X + self.bias
            error = y_pred - y

            # Compute gradients
            dw = (1/n) * np.dot(error, X)
            db = (1/n) * np.sum(error)

            # Update parameters
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return self.weight * X + self.bias
