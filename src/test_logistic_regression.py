import numpy as np
from logistic_regression import LogisticRegressionScratch

# Create a simple dataset
# Feature: [hours studied]
# Label: 1 = passed, 0 = failed

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)

model = LogisticRegressionScratch(learning_rate=0.1, epochs=3000)
model.fit(X, y)

preds = model.predict(X)
probs = model.predict_proba(X)

print("Weights:", model.weights)
print("Bias:", model.bias)

print("Probabilities:", probs)
print("Predictions:", preds)

accuracy = np.mean(preds == y)
print("Accuracy:", accuracy)