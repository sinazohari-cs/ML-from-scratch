import numpy as np

def generate_data(n=200):
    np.random.seed(0)

    # Class 0 (left cluster)
    x0 = np.random.randn(n//2, 2) + np.array([-2, -2])

    # Class 1 (right cluster)
    x1 = np.random.randn(n//2, 2) + np.array([2, 2])

    X = np.vstack((x0, x1))
    y = np.array([0]*(n//2) + [1]*(n//2))

    return X, y
# ---------------------------------------------------------
# Multiclass data generator for Softmax Regression
# ---------------------------------------------------------
def generate_multiclass_data(num_samples=600, num_classes=3, type="blobs", noise=0.2):
    """
    Generates a 2D dataset for multiclass classification.
    Supports:
        - blobs (clusters)
        - spiral (harder dataset)
    """

    if type == "blobs":
        centers = []
        for i in range(num_classes):
            angle = 2 * np.pi * i / num_classes
            radius = 5
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])

        centers = np.array(centers)
        samples_per_class = num_samples // num_classes

        X = []
        y = []

        for class_id in range(num_classes):
            cx, cy = centers[class_id]
            pts = np.random.randn(samples_per_class, 2) * noise + np.array([cx, cy])
            X.append(pts)
            y += [class_id] * samples_per_class

        return np.vstack(X), np.array(y)

    # ---------------------------------------------------------
    # Spiral dataset (more difficult)
    # ---------------------------------------------------------
    elif type == "spiral":
        X = np.zeros((num_samples, 2))
        y = np.zeros(num_samples, dtype='uint8')
        samples_per_class = num_samples // num_classes

        for class_id in range(num_classes):
            ix = range(class_id * samples_per_class, (class_id + 1) * samples_per_class)
            r = np.linspace(0.0, 1, samples_per_class)
            theta = (
                class_id * 4 +
                np.linspace(0.0, 4, samples_per_class) +
                np.random.randn(samples_per_class) * noise
            )
            X[ix] = np.c_[r * np.sin(theta), r * np.cos(theta)]
            y[ix] = class_id

        return X, y

    else:
        raise ValueError("type must be either 'blobs' or 'spiral'")

