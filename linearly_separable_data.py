import math
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    z = 1 / (1 + np.exp(-z))
    return z


def compute_cost(X, y, w, b):
    m, n = X.shape
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i) - (1 - y[i])*np.log( 1 - f_wb_i)

    cost = cost / m
    return cost


def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        err = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err*X[i, j]
        dj_db = dj_db + err

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(X, y, w_initial, b_initial, learning_rate, iterations):
    m, n = X.shape
    cost_history = []
    weights_history = []

    for i in range(iterations):
        # Calculate gradient
        dj_dw, dj_db = compute_gradient(X, y, w_initial, b_initial)

        # Update w and b after calculating gradient
        w_initial = w_initial - (learning_rate * dj_dw)
        b_initial = b_initial - (learning_rate * dj_db)

        cost = compute_cost(X, y, w_initial, b_initial)
        cost_history.append(cost)

        weights_history.append(w_initial)

        if i % math.ceil(iterations/10) == 0 or i == (iterations - 1):
            print(f"Iteration {i:4}: Cost {float(cost_history[-1]):8.2f}")

    return w_initial, b_initial, cost_history, weights_history


def predict(X, w, b):
    m, n = X.shape
    p = np.zeros(m)

    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb = sigmoid(z_i)

        if f_wb > 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p


def plot_decision_boundary(X_train, y_train, w_out, b_out):
    x = [point[0] for point in X_train]
    y = [point[1] for point in X_train]
    fig, ax = plt.subplots()

    for i, t in enumerate(y_train):
        if t == 1:
            ax.scatter(x[i], y[i], marker='x', color='red')
        else:
            ax.scatter(x[i], y[i], marker='o', color='blue')

    ax.axis([-4, 10, -4, 10])
    ax.set_ylabel('$x_1$', fontsize=12)
    ax.set_xlabel('$x_0$', fontsize=12)

    # calculate decision boundary
    # Define the x range to plot the line
    x = np.linspace(-10, 10, 100)

    # Solve for y in terms of x
    y = (1.88 * x + 1.55) / 2.39
    # Plot the line as a decision boundary
    plt.plot(x, y, 'k-', label='Decision Boundary')
    plt.show()


def cost_vs_iteration(cost_history):
    iterations_list = np.arange(30)
    plt.scatter(iterations_list, cost_history)
    plt.show()


X_train = np.array([[1, 1], [9.4, 6.4], [2.5, 2.1], [8, 7.7], [0.5, 2.2], [7.9, 8.4], [7, 7], [2.8, 0.8], [1.2, 3], [7.8, 6.1]])
y_train = [1, 1, 1, 0, 0, 0, 0, 1, 0, 1]

learning_rate = 0.14
iterations = 30
w_initial = np.array([-0.6, 0.75])
b_initial = 0.5

w, b, cost_history, weight_history = gradient_descent(X_train, y_train, w_initial, b_initial, learning_rate, iterations)
p = predict(X_train, np.array([1.88, -2.39]), 1.55)

print("Actual: ", y_train)
print("Predicted: ", p)
print(w, b)

plot_decision_boundary(X_train, y_train, w, b)
cost_vs_iteration(cost_history)
