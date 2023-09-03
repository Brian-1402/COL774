import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def abs_path(relative_pos):
    return os.path.join(os.path.dirname(__file__), relative_pos)


class SGDRegressor:
    def __init__(
        self,
        input_x=np.array([]),
        input_y=np.array([]),
        theta=np.array([]),
        actual_theta=np.array([]),
    ):
        """
        input_x: shape: (n,m)
        input_y: shape: (1,m)
        theta: shape: (n+1,1)
        If x and y values are not passed, assumed theta is given and data will be generated later.
        """

        self.X, self.y, self.theta, self.actual_theta, self.m = (
            input_x,
            input_y,
            theta,
            actual_theta,
            0,
        )
        if input_x.size > 0:
            self.n, self.m = input_x.shape
        elif theta.size > 0:
            self.n = theta.shape[0] - 1

    def generate_x(self, m, mean, std, noise):
        """Assumes theta and n is already initialized.
        mean: shape: (n,1)
        std: shape: (n,1)
        noise: (mean,std)
        """
        self.n, self.m = self.actual_theta.shape[0] - 1, m
        self.X = np.vstack((np.ones((1, m)), np.random.normal(mean, std, (self.n, m))))
        self.y = self.actual_theta.T @ self.X + np.random.normal(
            noise[0], noise[1], (1, m)
        )

    def J(self, X, y, theta=np.array([])):
        if theta.size == 0:
            theta = self.theta
        return ((theta.T @ X - y) ** 2).sum() / (2 * y.size)

    def stochastic_descent(self, r, eta=1e-3, limit=1e-5):
        X, y, m, self.theta_path = self.X.copy(), self.y, self.m, [self.theta.copy()]
        self.epoch_count = 1
        loss_diff = float("inf")
        while loss_diff > limit and self.epoch_count < 100:
            # np.random.shuffle(X.T)
            # print(f"Epoch {self.epoch_count}")
            perm = np.random.permutation(m)
            loss, batch_losses = self.J(X, y), []
            X, y = X.T[perm].T, y.T[perm].T  # shuffles
            for b in range(1, m // r + 1):
                # print(f" {b}", end="")

                B, y_b = X[:, (b - 1) * r : b * r], y[:, (b - 1) * r : b * r]
                del_J = B @ (self.theta.T @ B - y_b).T / r
                # print("del_J", del_J)
                self.theta -= del_J * eta

                self.theta_path.append(self.theta.copy())
                batch_losses.append(self.J(B, y_b))

            loss_new = np.average(batch_losses)
            # print(f"avg loss:{loss_new}")
            loss_diff = loss - loss_new
            # print("loss_diff:", loss_diff, "\n")
            loss = loss_new
            self.epoch_count += 1

    def plot_movement(self, r):
        # self.stochastic_descent(r=r)
        theta = np.array(self.theta_path).T[0]
        # print(theta[0, 1:])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("theta0")
        ax.set_ylabel("theta1")
        ax.set_zlabel("theta2")
        ax.set_title(f"Movement of theta for batch size {r}")
        ax.plot3D(theta[0], theta[1], theta[2], c="orange", label=f"b = {r}")
        ax.legend()
        plt.show()


s = SGDRegressor(
    theta=np.array([[0], [0], [0]], dtype="float64"),
    actual_theta=np.array([[3], [1], [2]], dtype="float64"),
)
means, stds = np.array([[3], [-1]]), np.array([[2], [2]])
noise = (0, 2**0.5)
s.generate_x(1_000_000, means, stds, noise)


q2test = pd.read_csv(abs_path("data\\q2\\q2test.csv"), header=0).to_numpy(
    dtype="float64"
)
X_test, y_test = (
    np.vstack((np.ones((1, q2test.shape[0])), q2test[:, 0:2].T)),
    q2test[:, 2].T,
)

for r, limit in [(1, 4e-2), (100, 4e-3), (10_000, 1e-5), (1_000_000, 1e-5)]:
    print(f"r: {r}, convergence criteria: {limit}")
    s.theta = np.array([[0], [0], [0]], dtype="float64")
    s.stochastic_descent(r=r, limit=limit)
    print(f"theta learnt: {s.theta.squeeze()}")
    print(f"Epochs: {s.epoch_count}")
    print(f"Test error: {s.J(X_test,y_test)}\n")
    s.plot_movement(r)

OG_test_error = s.J(X_test, y_test, s.actual_theta)
print(f"Test error with original hypothesis: {OG_test_error}")
