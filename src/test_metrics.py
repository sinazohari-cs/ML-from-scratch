import numpy as np
from linear_regression import LinearRegressionScratch
from metrics import mse, mae, r2_score

# Data
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

# Train model
model = LinearRegressionScratch(learning_rate=0.01, epochs=2000)
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Print metrics
print("MSE:", mse(y, y_pred))
print("MAE:", mae(y, y_pred))
print("R² Score:", r2_score(y, y_pred))