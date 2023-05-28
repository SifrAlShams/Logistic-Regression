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


def zscore_normalize_features(X):
    # find the mean of each column/feature
    # mu will have shape (n,)
    mu = np.mean(X, axis=0)

    # find the standard deviation of each column/feature
    # sigma will have shape (n,)
    sigma = np.std(X, axis=0)

    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


def cost_vs_iteration(cost_history, m):
    iterations_list = np.arange(m)
    plt.plot(iterations_list, cost_history)
    plt.show()


def plot_decision_boundary(X_train, y_train, w_out, b_out):
    def equation(x, y):
        return w_out[0] * x + w_out[1] * y + w_out[2] * x * y + w_out[3] * (x ** 2) + b_out

    x = [point[0] for point in X_train]
    y = [point[1] for point in X_train]
    fig, ax = plt.subplots()

    for i, t in enumerate(y_train):
        if t == 1:
            ax.scatter(x[i], y[i], marker='x', color='red')
        else:
            ax.scatter(x[i], y[i], marker='o', color='blue')

    # plot decision boundary
    # Define the range of x and y values
    x_values = np.linspace(0, 10, 100)
    y_values = np.linspace(0, 10, 100)

    # Create a meshgrid from the x and y values
    X, Y = np.meshgrid(x_values, y_values)

    # Evaluate the equation on the meshgrid
    Z = equation(X, Y)

    # Plot the contour of the equation
    plt.contour(X, Y, Z, levels=[0], colors='black')

    # Label the plot
    plt.title('4.16x - 9.53y - 13.05x^2 + 16.51xy - 0.7 = 0')
    plt.xlabel('x')
    plt.ylabel('y')

    # Show the plot
    plt.show()


X_train = np.array([[1, 1], [9.4, 6.4], [2.5, 2.1], [8, 7.7], [0.5, 2.2], [7.9, 8.4], [7, 7], [2.8, 0.8], [1.2, 3], [7.8, 6.1]])
y_train = [1, 0, 1, 0, 0, 1, 0, 1, 0, 0]

# Feature Engineering
x1 = []
x2 = []
for f_data in X_train:
    x1.append(f_data[0])
    x2.append(f_data[1])

x1 = np.array(x1)
x2 = np.array(x2)

X_Train = np.c_[x1, x2, x1*x2, x1**2]

# Feature Scaling
# X_norm, mu, sigma = zscore_normalize_features(X_Train)

# Run Gradient Descent
learning_rate = 0.003
iterations = 18000
w_initial = np.array([0.0, 0.0, 0.0, 0.0])
b_initial = 0.0

w, b, cost_history, weight_history = gradient_descent(X_Train, y_train, w_initial, b_initial, learning_rate, iterations)
p = predict(X_Train, w, b)
print(w, b)
print("Actual:", y_train)
print("Predicted:", p)
cost_vs_iteration(cost_history, iterations)
plot_decision_boundary(X_train, y_train, w, b)