import numpy as np
from linear_regression import LinearRegressionScratch

# Fake dataset
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

model = LinearRegressionScratch(learning_rate=0.02, epochs=5000)
model.fit(X, y)

print("Weight:", model.weight)
print("Bias:", model.bias)

print("Prediction for x=6:", model.predict(np.array([6], dtype=float)))
