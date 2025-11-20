import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegressionScratch

# Data
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

# Train model
model = LinearRegressionScratch(learning_rate=0.01, epochs=2000)
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Plot
plt.figure(figsize=(8, 5))

# Scatter points
plt.scatter(X, y, label="Data Points")

# Regression line
plt.plot(X, y_pred, label="Fitted Line")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression From Scratch")
plt.legend()

plt.show()
