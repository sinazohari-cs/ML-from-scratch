import numpy as np


class SoftmaxRegression:
    """
    Multiclass Logistic Regression (Softmax Regression)
    implemented from scratch using NumPy.
    """

    def __init__(self, num_features, num_classes, lr=0.1):
        self.num_features = num_features
        self.num_classes = num_classes
        self.lr = lr

        # Initialize weight matrix (features × classes)
        self.W = np.random.randn(num_features, num_classes) * 0.01
        self.b = np.zeros((1, num_classes))

    # -------------------------------------------------------
    # Utility: Softmax function
    # -------------------------------------------------------
    def softmax(self, z):
        # subtract max for numerical stability
        z -= np.max(z, axis=1, keepdims=True)
        exp_scores = np.exp(z)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # -------------------------------------------------------
    # Predict class probabilities
    # -------------------------------------------------------
    def predict_proba(self, X):
        z = np.dot(X, self.W) + self.b
        return self.softmax(z)

    # -------------------------------------------------------
    # Predict class labels (argmax)
    # -------------------------------------------------------
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    # -------------------------------------------------------
    # One-hot encoding
    # -------------------------------------------------------
    def one_hot(self, y):
        onehot = np.zeros((len(y), self.num_classes))
        onehot[np.arange(len(y)), y] = 1
        return onehot

    # -------------------------------------------------------
    # Cross-entropy loss
    # -------------------------------------------------------
    def compute_loss(self, X, y):
        m = X.shape[0]
        probs = self.predict_proba(X)
        y_onehot = self.one_hot(y)

        loss = -np.sum(y_onehot * np.log(probs + 1e-9)) / m
        return loss

    # -------------------------------------------------------
    # Training loop using gradient descent
    # -------------------------------------------------------
    def fit(self, X, y, epochs=2000, verbose=True):
        m = X.shape[0]
        y_onehot = self.one_hot(y)

        for epoch in range(epochs):
            probs = self.predict_proba(X)

            # Gradients
            dW = np.dot(X.T, (probs - y_onehot)) / m
            db = np.sum(probs - y_onehot, axis=0, keepdims=True) / m

            # Update parameters
            self.W -= self.lr * dW
            self.b -= self.lr * db

            if verbose and epoch % 200 == 0:
                loss = self.compute_loss(X, y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
