# Plots are activated with the plt.show() command
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def abs_path(relative_pos):
    return os.path.join(os.path.dirname(__file__), relative_pos)


def normalize(arr):
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    return (arr - mean) / std, mean, std


X = pd.read_csv(abs_path("data/q4/q4x.dat"), header=None, sep="\s+").to_numpy()
Y = pd.read_csv(abs_path("data/q4/q4y.dat"), header=None, sep="\s+").to_numpy()
X = X.astype("float64")

X_norm, X_mean, X_std = normalize(X)


def execute_Q4_a(X, Y):
    phi = np.sum([Y.T[0] == "Alaska"]) / len(Y.T[0])
    mu_0 = np.array(
        [np.mean(X.T[0][Y.T[0] == "Alaska"]), np.mean(X.T[1][Y.T[0] == "Alaska"])]
    )
    mu_1 = np.array(
        [np.mean(X.T[0][Y.T[0] == "Canada"]), np.mean(X.T[1][Y.T[0] == "Canada"])]
    )
    S1 = np.array(
        [X.T[0][Y.T[0] == "Alaska"] - mu_0[0], X.T[1][Y.T[0] == "Alaska"] - mu_0[1]]
    )
    S2 = np.array(
        [X.T[0][Y.T[0] == "Canada"] - mu_1[0], X.T[1][Y.T[0] == "Canada"] - mu_1[1]]
    )
    S = (S1 @ S1.T + S2 @ S2.T) / len(Y.T[0])
    return phi, mu_0, mu_1, S


def execute_Q4_b(X, Y):
    plt.scatter(
        X.T[0][Y.T[0] == "Alaska"],
        X.T[1][Y.T[0] == "Alaska"],
        color="Blue",
        label="Alaska",
    )
    plt.scatter(
        X.T[0][Y.T[0] == "Canada"],
        X.T[1][Y.T[0] == "Canada"],
        color="Orange",
        label="Canada",
    )
    plt.legend(loc="upper right")
    plt.xlabel("x1")
    plt.ylabel("x2")


def execute_Q4_c(X, Y):
    execute_Q4_b(X, Y)
    phi, mu_0, mu_1, S = execute_Q4_a(X, Y)
    S_inv = np.linalg.inv(S)
    c = -np.log(phi / (1 - phi)) - 0.5 * (mu_1.T @ S_inv @ mu_1 - mu_0.T @ S_inv @ mu_0)
    a = -(mu_1 - mu_0).T @ S_inv
    x = np.arange(-3, 4)
    y = -1 * (c + a[0] * x) / a[1]
    plt.plot(x, y, color="green")


def execute_Q4_d(X, Y):
    phi, mu_0, mu_1, p = execute_Q4_a(X, Y)
    S1 = np.array(
        [X.T[0][Y.T[0] == "Alaska"] - mu_0[0], X.T[1][Y.T[0] == "Alaska"] - mu_0[1]]
    )
    S2 = np.array(
        [X.T[0][Y.T[0] == "Canada"] - mu_1[0], X.T[1][Y.T[0] == "Canada"] - mu_1[1]]
    )
    return (S1 @ S1.T) / np.sum([Y.T[0] == "Alaska"]), (S2 @ S2.T) / np.sum(
        [Y.T[0] == "Canada"]
    )


def execute_Q4_e(X, Y):
    x0 = np.linspace(-2, 2, 50)
    phi, mu_0, mu_1, S = execute_Q4_a(X, Y)
    S0, S1 = execute_Q4_d(X, Y)
    S0_inv = np.linalg.inv(S0)
    S1_inv = np.linalg.inv(S1)
    lin = mu_1.T @ S1_inv - mu_0.T @ S0_inv
    a = (S1_inv - S0_inv).flatten()[3]
    b = (
        (S1_inv - S0_inv).flatten()[1] + (S1_inv - S0_inv).flatten()[2]
    ) * x0 - 2 * lin[1]
    c = (
        (S1_inv - S0_inv).flatten()[0] * (x0**2)
        - 2 * lin[0] * x0
        + 2
        * np.log(
            ((1 - phi) / (phi) * (np.linalg.det(S1) ** 0.5 / np.linalg.det(S0) ** 0.5))
        )
        + 1 * (mu_1.T @ S1_inv @ mu_1 - mu_0.T @ S0_inv @ mu_0)
    )
    x1 = (-b + np.sqrt((b**2) - 4 * a * c)) / (2 * a)
    plt.plot(x0, x1, color="black")


def execute_Q4_f(X, Y):
    execute_Q4_c(X, Y)
    execute_Q4_e(X, Y)


# Q4_a = execute_Q4_a(X_norm, Y)
# print(f"phi: {Q4_a[0]}, mu0: {Q4_a[1]}, mu1: {Q4_a[2]}, S: {Q4_a[3]}")

# execute_Q4_b(X_norm, Y)
# execute_Q4_c(X_norm, Y)
# execute_Q4_d(X_norm, Y)
# execute_Q4_e(X_norm, Y)
execute_Q4_f(X_norm, Y)
plt.show()
