# ML-from-Scratch

A machine learning library I built from scratch using only **NumPy**.

The main goal of this project was to better understand how machine learning algorithms actually work under the hood instead of relying on libraries like scikit-learn. Every model was implemented manually, including the training process, loss calculations, gradient updates, and prediction logic.

While building this project, I focused on writing everything myself—from the mathematical foundations to the optimization routines—to get a deeper understanding of the algorithms.

## What’s Included

### Linear Regression

* Gradient Descent training
* Mean Squared Error (MSE) loss
* Regression line visualization

### Logistic Regression

* Sigmoid activation
* Binary Cross-Entropy loss
* Classification predictions
* Decision boundary visualization
* Training loss plots

### Softmax Regression

* Multiclass classification
* Softmax activation
* Cross-Entropy loss
* Probability-based predictions

### Decision Tree Classifier

* Gini impurity splitting criterion
* Recursive tree construction
* Maximum depth and minimum sample constraints
* Tree traversal for predictions
* Implemented entirely with NumPy

## Project Structure

Each model has its own test file located in the `src` directory:

```bash
test_linear_regression.py
test_logistic_regression.py
test_softmax_regression.py
test_decision_tree.py
test_metrics.py
```

To run a test:

```bash
python src/test_linear_regression.py
```

## Why I Built This

I created this project as a learning exercise to strengthen my understanding of machine learning fundamentals. Building these algorithms from scratch helped me understand concepts like gradient descent, optimization, loss functions, decision boundaries, and tree-based learning at a much deeper level than simply using existing libraries.

This project is still growing, and I plan to continue adding more algorithms and improvements over time.
