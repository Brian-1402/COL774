import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def abs_path(relative_pos):
    return os.path.join(os.path.dirname(__file__), relative_pos)


linearX = pd.read_csv(abs_path("data\\q1\\linearX.csv"), header=None).to_numpy()
linearY = pd.read_csv(abs_path("data\\q1\\linearY.csv"), header=None).to_numpy()


def normalize(arr):
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    return (arr - mean) / std, mean, std


def linear_regression(
    input_x, input_y, theta=np.zeros((2, 1)), eta=1e-1, J_limit=1e-15
):
    """
    input_x: shape: (m,n), 0<m,n
    input_y: shape: (m,1)
    theta: shape: (n+1,1)

    """
    m = input_x.shape[0]
    X, y = np.concatenate((input_x, np.ones((m, 1))), axis=1), input_y

    cost_diff = float("inf")
    iter_count = 0

    while cost_diff > J_limit:
        del_J = X.T @ (X @ theta - y) / m
        prev_cost = ((X @ theta - y) ** 2).sum() / (2 * m)
        theta -= del_J * eta
        cost_diff = prev_cost - ((X @ theta - y) ** 2).sum() / (2 * m)
        iter_count += 1

    return theta, iter_count

    # print(theta, del_J, end="\n\n\n")


def predict(input_x, theta, mean=None, std=None):
    """
    input_x: shape: (m,n), 0<m,n
    theta: shape: (n+1,1)
    """
    if mean != None:
        input_x = (input_x - mean) / std

    m = input_x.shape[0]
    X = np.concatenate((input_x, np.ones((m, 1))), axis=1)

    return X @ theta


def execute_Q1_part_a():
    X_norm, X_mean, X_std = normalize(linearX)
    m = linearX.shape[0]
    eta = 1e-1

    theta, iter_count = linear_regression(X_norm, linearY, eta=eta)

    print(f"number of iterations: {iter_count}\neta:{eta}\n")
    print(f"final theta: {theta[0]}, {theta[1]}")


def execute_Q1_part_b():
    input_x = normalize(linearX)[0]
    theta = linear_regression(input_x, linearY)[0]
    h = predict(input_x, theta)

    points = plt.scatter(linearX, linearY, color="b")
    line = plt.plot(linearX, h, color="r")

    plt.xlabel("Acidity")
    plt.ylabel("Density")
    plt.show()


execute_Q1_part_b()
