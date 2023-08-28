import os
import pandas as pd
import numpy as np


def abs_path(relative_pos):
    return os.path.join(os.path.dirname(__file__), relative_pos)


linearX = pd.read_csv(abs_path("data\\q1\\linearX.csv")).to_numpy()  # (99,1)
linearY = pd.read_csv(abs_path("data\\q1\\linearY.csv")).to_numpy()

# linearX = np.concatenate((linearX, np.ones((99, 1))), axis=1)
# add x0 values after normalizing, and just before you do descent,
# otherwise normalizing a set of 1s will become division by zero

theta = np.zeros((2, 1))  # (2,1)


def normalize(arr):
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    return (arr - mean) / std, mean, std


X_norm, X_std, X_mean = normalize(linearX)
m = linearX.shape[0]

X, y = np.concatenate((X_norm, np.ones((99, 1))), axis=1), linearY
# X = (99,2), y - (99,1)


# starting gradient descent
# del_J = float("inf")
eta = 1e-1

del_J = X.T @ (X @ theta - y) / m
cost_diff = float("inf")
iter_count = 0

while cost_diff > 1e-15:
    del_J = X.T @ (X @ theta - y) / m
    prev_cost = ((X @ theta - y) ** 2).sum() / m
    theta -= del_J * eta
    cost_diff = prev_cost - ((X @ theta - y) ** 2).sum() / m
    iter_count += 1

    # print(theta, del_J, end="\n\n\n")

print(f"number of iterations: {iter_count}\neta:{eta}\n")
print(f"final theta: {theta[0]}, {theta[1]}")
