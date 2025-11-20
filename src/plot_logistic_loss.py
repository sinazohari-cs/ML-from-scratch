import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegressionScratch

# Dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)

# Train model
model = LogisticRegressionScratch(learning_rate=0.1, epochs=3000)
model.fit(X, y)

# Plot loss
plt.plot(model.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Logistic Regression Loss Curve')
plt.show()
