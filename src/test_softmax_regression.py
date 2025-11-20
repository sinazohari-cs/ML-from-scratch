import numpy as np
from generate_2d_data import generate_multiclass_data
from softmax_regression import SoftmaxRegression


# ---------------------------------------------------------
# Generate a 3-class spiral or blob dataset
# ---------------------------------------------------------
X, y = generate_multiclass_data(
    num_samples=600,
    num_classes=3,
    type="blobs",     # choose: "blobs" or "spiral"
    noise=0.2
)

num_features = X.shape[1]
num_classes = len(np.unique(y))

# ---------------------------------------------------------
# Create & train model
# ---------------------------------------------------------
model = SoftmaxRegression(
    num_features=num_features,
    num_classes=num_classes,
    lr=0.1
)

model.fit(X, y, epochs=1500, verbose=True)

# ---------------------------------------------------------
# Print accuracy
# ---------------------------------------------------------
preds = model.predict(X)
accuracy = np.mean(preds == y)
print(f"\nTraining Accuracy: {accuracy * 100:.2f}%")

# ---------------------------------------------------------
# Print sample predictions
# ---------------------------------------------------------
print("\nExample Predictions:")
for i in range(5):
    print(f"  X={X[i]},  True={y[i]},  Pred={preds[i]}")
