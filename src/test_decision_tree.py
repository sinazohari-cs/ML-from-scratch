import numpy as np
from decision_tree import DecisionTree

# Generate synthetic dataset
np.random.seed(0)
X = np.random.rand(200, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # label = 1 if x+y > 1 else 0

# Train
tree = DecisionTree(max_depth=4)
tree.fit(X, y)

# Predictions
preds = tree.predict(X)
accuracy = np.mean(preds == y) * 100

print(f"Training Accuracy: {accuracy:.2f}%")
print("Example Predictions:")
for i in range(5):
    print(f"  X={X[i]}, True={y[i]}, Pred={preds[i]}")
