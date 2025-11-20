# ML-from-Scratch

A complete **machine learning library built from scratch**, using only **NumPy**.  
This project implements fundamental ML algorithms step-by-step without relying on scikit-learn or other machine-learning frameworks.

The goal is to deeply understand how core ML models work internally by manually writing:
- Forward passes  
- Loss functions  
- Gradients  
- Training loops  
- Metrics  
- Data generation  
- Visualizations  

Perfect for study, research, and portfolio demonstration.

---

## 🚀 Features

### ✔️ Linear Models
- **Linear Regression**  
  - Gradient Descent Optimization  
  - Mean Squared Error Loss  
  - Plotting fitted regression line

- **Logistic Regression**  
  - Sigmoid function  
  - Binary cross-entropy loss  
  - Decision boundary visualization  
  - Loss curve plot

- **Softmax Regression (Multiclass Logistic Regression)**  
  - Softmax function  
  - Cross-entropy loss  
  - Multiclass predictions

---

### ✔️ Tree Models
- **Decision Tree Classifier**  
  - Gini impurity  
  - Recursive splitting  
  - Max depth + min samples constraints  
  - Pure NumPy implementation  
  - Predicts by traversing the tree

---

## 🧪 Tests

Each model includes its own test file under `/src`:

- `test_linear_regression.py`
- `test_logistic_regression.py`
- `test_softmax_regression.py`
- `test_decision_tree.py`
- `test_metrics.py`

To run a test:

```bash
python src/test_linear_regression.py
