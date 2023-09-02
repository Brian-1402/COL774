import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def abs_path(relative_pos):
    return os.path.join(os.path.dirname(__file__), relative_pos)


class LogisticClassifier:
    def __init__(self, input_x, input_y, theta=np.array([])):
        """
        input_x: shape: (n,m)
        input_y: shape: (1,m)
        theta: shape: (n+1,1)
        """
        self.X, self.y, self.theta = input_x, input_y, theta
        self.X_mean, self.X_std = None, None
        self.n, self.m = self.X.shape
        if theta.size == 0:
            self.theta = np.zeros((self.n + 1, 1))
        self.normalize()
        self.X = np.concatenate((np.ones((1, self.m)), self.X), axis=-2)

    def reset_training_data(self, input_x, input_y, theta=None):
        self.X, self.y = input_x, input_y
        self.normalize()
        self.X = np.concatenate((np.ones((self.m, 1)), self.X), axis=-2)
        if theta != None:
            self.theta = theta

    def normalize(self):
        self.X_mean, self.X_std = (
            self.X.mean(axis=-1)[:, np.newaxis],
            self.X.std(axis=-1)[:, np.newaxis],
        )
        self.X = (self.X - self.X_mean) / self.X_std

    def LL(self, h=np.array([])):
        if h.size == 0:
            h = 1 / (1 + np.exp(-1 * (self.theta.T @ self.X)))  # (1,m)
        return self.y @ np.log(h).T + (1 - self.y) @ np.log(1 - h).T

    def train_newton(self, limit=1e-15):
        X, y, m = self.X, self.y, self.m
        LL_diff = float("inf")
        self.theta_steps, self.LL_steps = [self.theta.copy()], [self.LL()]
        exp_tx = np.exp(-1 * (self.theta.T @ X))
        h = 1 / (1 + exp_tx)  # (1,m)
        while LL_diff > limit:
            del_LL = (X @ (y - h).T) / m  # (n,1), maintaining vector style
            mid = np.diag((-1 * exp_tx * h * h).squeeze())  # (m,m)
            Hessian_inv = np.linalg.inv((X @ mid @ X.T) / m)  # (n,n)

            self.theta -= Hessian_inv @ del_LL

            exp_tx = np.exp(-1 * (self.theta.T @ X))
            h = 1 / (1 + exp_tx)  # (1,m)
            LL_new = self.LL(h)
            LL_diff = LL_new - self.LL_steps[-1]
            self.LL_steps.append(LL_new)


logisticX = pd.read_csv(abs_path("data\\q3\\logisticX.csv"), header=None).to_numpy().T
logisticY = pd.read_csv(abs_path("data\\q3\\logisticY.csv"), header=None).to_numpy().T

logistic = LogisticClassifier(logisticX, logisticY)
logistic.train_newton()

# l_y = np.broadcast_to(logisticY, (logisticX.shape))

points_1 = np.compress(logisticY.squeeze() == 1, logisticX, axis=-1)
points_0 = np.compress(logisticY.squeeze() == 0, logisticX, axis=-1)

plt.scatter(points_1[0, :], points_1[1, :], marker="x")
plt.scatter(points_0[0, :], points_0[1, :], marker="o")
x = np.array([1, 9])
y = (-logistic.theta[0] - logistic.theta[1] * x) / logistic.theta[2]
plt.plot(x, y)
plt.show()
