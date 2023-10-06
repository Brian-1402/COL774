import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cvxopt import matrix, solvers
from datetime import datetime
from numba import jit

now = datetime.now


def abs_path(relative_pos):
    return os.path.join(os.path.dirname(__file__), relative_pos)


def process_img(img):
    img = img.resize((16, 16))
    img_arr = np.array(img, dtype="float64")
    img_arr /= 255
    return img_arr.flatten()


"""Assembling data"""
train_p, train_n = [], []
val_p, val_n = [], []

for f in os.scandir(abs_path("data/svm/train/3")):
    img = Image.open(f.path)
    train_p.append(process_img(img))
for f in os.scandir(abs_path("data/svm/train/4")):
    img = Image.open(f.path)
    train_n.append(process_img(img))

for f in os.scandir(abs_path("data/svm/val/3")):
    img = Image.open(f.path)
    val_p.append(process_img(img))
for f in os.scandir(abs_path("data/svm/val/4")):
    img = Image.open(f.path)
    val_n.append(process_img(img))

train_p, train_n = np.stack(train_p), np.stack(train_n)
val_p, val_n = np.stack(val_p), np.stack(val_n)

train_x = np.concatenate([train_p, train_n])
m = train_x.shape[0]
train_y = np.concatenate(
    [np.ones((train_p.shape[0], 1)), -1 * np.ones((train_n.shape[0], 1))]
)
val_x = np.concatenate([val_p, val_n])
val_y = np.concatenate(
    [np.ones((val_p.shape[0], 1)), -1 * np.ones((val_n.shape[0], 1))]
)

"""Generate CVXOPT inputs"""


class SVM:
    def __init__(self, C=1, kernel="linear"):
        # self.train_x = train_x
        # self.train_y = train_y
        self.C = C
        kernel_map = {"linear": self.K_linear}
        self.K = kernel_map[kernel]

    def K_linear(self, X1, X2):
        return np.dot(X1, X2.T)

    def train(self, train_x, train_y):
        C = self.C
        print("Generating CVXOPT params")
        t = now()

        P = np.ones((m, m))
        for i in range(m):
            for j in range(m):
                P[i, j] *= (
                    train_y[i, 0] * train_y[j, 0] * self.K(train_x[i], train_x[j])
                )
        q = -1 * np.ones(m)
        G = np.concatenate([-1 * np.identity(m), np.identity(m)])
        h = np.concatenate([np.zeros(m), C * np.ones(m)])
        A = train_y.reshape((1, train_y.size))
        b = np.array([[0.0]]).reshape((1, 1))

        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)
        print("time taken:", now() - t, "\n")

        print("\nCVXOPT Solving")
        t = now()
        sol = solvers.qp(P, q, G, h, A, b)
        sols = np.array(sol["x"])
        print("time taken:", now() - t, "\n")

        sols.dump("sols.npy")
        sols = np.load("sols.npy", allow_pickle=True)

        sv_pos = sols.flatten() > 1e-6
        self.m_sv = np.count_nonzero(sv_pos)  # 3066
        self.alpha = sols[sv_pos].reshape((self.m_sv, 1))  # (m,1)
        self.X_sv = train_x[sv_pos, :]  # (m,768)
        self.y_sv = train_y[sv_pos, :]  # (m,1)
        self.b = np.mean(self.y_sv - self.wx(self.X_sv))  # -0.7401936390874317

    def train_precomputed(self, sols):
        sv_pos = sols.flatten() > 1e-6
        self.m_sv = np.count_nonzero(sv_pos)  # 3066
        self.alpha = sols[sv_pos].reshape((self.m_sv, 1))  # (m,1)
        self.X_sv = train_x[sv_pos, :]  # (m,768)
        self.y_sv = train_y[sv_pos, :]  # (m,1)

        b = np.mean(self.y_sv - self.wx(self.X_sv))  # -0.7401936390874317
        # p, n = self.y_sv.flatten() == 1, self.y_sv.flatten() == -1
        # b = -0.5 * np.max(self.wx(self.X_sv[n, :])) + np.min(self.wx(self.X_sv[p, :]))
        # # -0.6591808870930791
        self.b = b

    def wx(self, X_test):
        # X - (m,n), X_test - (k,n)
        # a,y - (1,m)
        w_x = (self.alpha * self.y_sv * self.K(self.X_sv, X_test)).sum(axis=0)
        return w_x.reshape((w_x.size, 1))

    def predict(self, X_test):
        return 2.0 * (self.wx(X_test) + self.b > 0) - 1

    def test(self, X_test, y_test):
        ans = self.predict(X_test) == y_test
        print(f"Accuracy: {100 * np.mean(ans) :.2f}%")


# sols = np.load("solstice.npz", allow_pickle=True)
sols = np.load("sols.npy", allow_pickle=True)
s = SVM()
s.train(sols)
s.test(val_x, val_y)
