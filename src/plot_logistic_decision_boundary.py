import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegressionScratch
from generate_2d_data import generate_data

# Load data
X, y = generate_data(200)

# Train model
model = LogisticRegressionScratch(learning_rate=0.1, epochs=5000)
model.fit(X, y)

# Plot data points
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', alpha=0.7)

# Create decision boundary
x1_range = np.linspace(X[:,0].min(), X[:,0].max(), 100)
x2_range = np.linspace(X[:,1].min(), X[:,1].max(), 100)
xx1, xx2 = np.meshgrid(x1_range, x2_range)
grid = np.c_[xx1.ravel(), xx2.ravel()]

probs = model.predict_proba(grid).reshape(xx1.shape)

# Draw contour
plt.contourf(xx1, xx2, probs, levels=[0, 0.5, 1], alpha=0.2, cmap='bwr')

plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()
