# Logistic Regression on Linear vs. Non-Linear Data

This project implements logistic regression models to classify:

1. **Linearly separable** data using basic features.
2. **Non-linearly separable** data using polynomial feature engineering.

## Methods & Architecture

### 1. `linearly_separable_data.py`

* Uses raw input features $[x_1, x_2]$.
* Trains logistic regression via **batch gradient descent**.
* Plots a linear decision boundary.

### 2. `linearly_non_separable_data.py`

* Engineers non-linear features: $[x_1, x_2, x_1x_2, x_1^2]$.
* Trains logistic regression on transformed input.
* Plots a **non-linear decision boundary** via contouring.

## Key Functions

* `sigmoid(z)`: Logistic activation function
* `compute_cost(X, y, w, b)`: Binary cross-entropy loss
* `compute_gradient(X, y, w, b)`: Derivatives of cost w\.r.t weights and bias
* `gradient_descent(...)`: Parameter updates
* `predict(X, w, b)`: Binary predictions
* `plot_decision_boundary(...)`: Decision boundary visualization
* `cost_vs_iteration(...)`: Training loss visualization

## Data

Synthetic 2D data for binary classification.

```python
X_train = [[1,1], [9.4,6.4], [2.5,2.1], [8,7.7], [0.5,2.2],
           [7.9,8.4], [7,7], [2.8,0.8], [1.2,3], [7.8,6.1]]
```

