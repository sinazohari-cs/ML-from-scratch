import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# 1. Create Fake Data
# ----------------------
np.random.seed(0)

X_class0 = np.random.randn(50, 2) + np.array([-2, -2])
X_class1 = np.random.randn(50, 2) + np.array([2, 2])

X = np.vstack([X_class0, X_class1])
y = np.array([0]*50 + [1]*50).reshape(-1, 1)

# ----------------------
# 2. Helper Functions
# ----------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, weights, bias):
    return sigmoid(np.dot(X, weights) + bias)

# ----------------------
# 3. Train Logistic Regression
# ----------------------
weights = np.zeros((2, 1))
bias = 0
lr = 0.1
epochs = 2000

loss_history = []

for i in range(epochs):
    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)

    loss = -np.mean(y*np.log(y_pred + 1e-8) + (1-y)*np.log(1-y_pred + 1e-8))
    loss_history.append(loss)

    dw = np.dot(X.T, (y_pred - y)) / len(X)
    db = np.mean(y_pred - y)

    weights -= lr * dw
    bias -= lr * db

# ----------------------
# 4. Accuracy
# ----------------------
y_hat = (predict(X, weights, bias) > 0.5).astype(int)
accuracy = np.mean(y_hat == y)
print("Accuracy:", accuracy)

# ----------------------
# 5. Plot Loss Curve
# ----------------------
plt.plot(loss_history)
plt.title("Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# ----------------------
# 6. Visualize Decision Boundary
# ----------------------
plt.scatter(X_class0[:,0], X_class0[:,1], label="Class 0")
plt.scatter(X_class1[:,0], X_class1[:,1], label="Class 1")

x_values = np.linspace(-5, 5, 100)
y_values = -(weights[0]*x_values + bias) / weights[1]

plt.plot(x_values, y_values, color="black", label="Decision Boundary")
plt.legend()
plt.show()
