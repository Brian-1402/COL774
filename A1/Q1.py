import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def abs_path(relative_pos):
    return os.path.join(os.path.dirname(__file__), relative_pos)


def normalize(arr):
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    return (arr - mean) / std, mean, std


linearX = pd.read_csv(abs_path("data\\q1\\linearX.csv"), header=None).to_numpy()
linearY = pd.read_csv(abs_path("data\\q1\\linearY.csv"), header=None).to_numpy()

X_norm, X_mean, X_std = normalize(linearX)


def linear_regression(
    input_x, input_y, theta=np.zeros((2, 1)), eta=1e-1, J_limit=1e-15
):
    """
    input_x: shape: (m,n), 0<m,n
    input_y: shape: (m,1)
    theta: shape: (n+1,1)

    """
    m = input_x.shape[0]
    X, y = np.concatenate((input_x, np.ones((m, 1))), axis=-1), input_y

    cost_diff = float("inf")
    iter_count = 0
    # cost_steps = np.append(theta.squeeze(), (y**2).sum() / (2 * m))[np.newaxis]
    cost_steps = [[np.squeeze(theta.copy()), (y**2).sum() / (2 * m)]]
    while cost_diff > J_limit:
        del_J = X.T @ (X @ theta - y) / m
        prev_cost = ((X @ theta - y) ** 2).sum() / (2 * m)
        theta -= del_J * eta
        new_cost = ((X @ theta - y) ** 2).sum() / (2 * m)
        cost_diff = prev_cost - new_cost
        iter_count += 1
        # step = np.append(theta.squeeze(), new_cost)
        # cost_steps = np.concatenate((cost_steps, step[np.newaxis]), axis=0)
        cost_steps.append([np.squeeze(theta.copy()), new_cost])

    return theta, iter_count, cost_steps

    # print(theta, del_J, end="\n\n\n")


def predict(input_x, theta, mean=None, std=None):
    """
    input_x: shape: (m, n), 0<m,n
    theta: shape: (..., n+1, 1)
    Supports broadcasting of X and theta
    """
    if mean != None:
        input_x = (input_x - mean) / std

    m = input_x.shape[0]
    X = np.concatenate((input_x, np.ones((m, 1))), axis=-1)
    # Axis -1 because we need to add 1s to make last dimension n to n+1.
    # For the 1s to be multiplied with theta_0, the intercept terms.

    return X @ theta


def execute_Q1_a():
    m = linearX.shape[0]
    eta = 1e-1

    theta, iter_count = linear_regression(X_norm, linearY, eta=eta)[:2]

    print(f"number of iterations: {iter_count}\neta:{eta}\n")
    print(f"final theta: {theta[0]}, {theta[1]}")


# [0.0013402], [0.99662001]


def execute_Q1_b():
    theta = linear_regression(X_norm, linearY)[0]
    plot_ends = np.array([[2.5], [17.5]])
    h = predict(plot_ends, theta, X_mean, X_std)

    plt.scatter(linearX, linearY, color="b")
    plt.plot(plot_ends, h, color="r")

    plt.xlabel("Acidity")
    plt.ylabel("Density")
    plt.show()


def execute_Q1_c():
    # -- For plotting the mesh graph of cost function --

    # Uses 2.7GB of RAM to display to granularity of 6e-3
    X = np.arange(-1, 1, 6e-3)  # (x,)
    Y = np.arange(0, 2, 6e-3)  # (y,)
    X, Y = np.meshgrid(X, Y)  # (y,x)
    # values of X only vary across x dimension, rest are copies, similarly for Y

    # thetas = np.concatenate((X[np.newaxis], Y[np.newaxis]), axis=0)  # (n+1, y, x)
    # It's like sticking the two layers together like a sandwich.
    # new_thetas = np.moveaxis(thetas, 0, -1)[:, :, :, np.newaxis]
    # (y, x, n+1, 1), think of it as a y*x grid of (n+1 x 1) theta vectors.
    # Alternatively,

    y, x = X.shape
    new_thetas = np.concatenate(
        (X.reshape((y, x, 1, 1)), Y.reshape((y, x, 1, 1))), axis=-2
    )
    # Concatenated the two sets of theta values along the 2nd last axis to make it look like:
    # (y, x, n+1, 1), think of it as a y*x grid of (n+1 x 1) theta vectors.

    prediction = predict(X_norm, new_thetas)
    # (y,x,m), in predict() we do matrix multiplication of (m,n+1) X_norm with (y,x,n+1,1),
    # with X_norm getting broadcasted and prediction evaluated as (y,x,m,1).

    J = ((prediction - linearY) ** 2).sum(axis=(-1, -2)) / (2 * linearX.shape[0])
    # (y,x). we sum over axes -1 and -2, because the elements to be added are in -2 axes,
    # and we include -1 as well just to eliminate that dimension,
    # because summing over a dimension of size 1 does no addition, and just removes that dimension.

    from matplotlib import cm

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, J, cmap=cm.coolwarm, antialiased=False, zorder=0)

    # -- For plotting the gradient descent path --
    steps = linear_regression(X_norm, linearY)[2]
    t0path = np.array([i[0] for i in steps])[:, 0]
    t1path = np.array([i[0] for i in steps])[:, 1]
    zpath = np.array([i[1] for i in steps])
    ax.plot(t0path, t1path, zpath, color="g", linewidth=2.0, zorder=10)
    plt.show()


# def execute_Q1_c_loops():
#     theta_0 = np.arange(-5, 5, 0.25)  # (20,)
#     theta_1 = np.arange(-5, 5, 0.25)  # (40,)
#     Z = np.zeros((theta_1.shape + theta_0.shape))
#     for t0 in range(len(theta_0)):
#         for t1 in range(len(theta_1)):
#             theta = np.array([theta_0[t0], theta_1[t1]]).reshape((2, 1))
#             m = X_norm.shape[0]
#             X = np.concatenate((X_norm, np.ones((m, 1))), axis=1)
#             J = ((X @ theta - linearY) ** 2).sum() / (2 * m)
#             Z[t1][t0] = J
#     X, Y = np.meshgrid(theta_0, theta_1)
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#     surf = ax.plot_surface(X, Y, Z, antialiased=False)
#     plt.show()


execute_Q1_c()
# -- For plotting the gradient descent path --
# steps = linear_regression(X_norm, linearY)[2]
# t0path = np.array([i[0] for i in steps])[:, 0]
# t1path = np.array([i[0] for i in steps])[:, 1]
# zpath = np.array([i[1] for i in steps])
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot(t0path, t1path, zpath, antialiased=False)

# plt.show()
