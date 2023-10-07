import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cvxopt import matrix, solvers
from datetime import datetime


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

for f in os.scandir(abs_path("data/svm/train/1")):
    img = Image.open(f.path)
    train_p.append(process_img(img))
for f in os.scandir(abs_path("data/svm/train/2")):
    img = Image.open(f.path)
    train_n.append(process_img(img))

for f in os.scandir(abs_path("data/svm/val/1")):
    img = Image.open(f.path)
    val_p.append(process_img(img))
for f in os.scandir(abs_path("data/svm/val/2")):
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
    def __init__(self, kernel="linear", C=1, g_lambda=1e-3):
        self.C = C
        kernel_map = {"linear": self.K_linear, "gaussian": self.K_gaussian}
        self.K = kernel_map[kernel]
        self.g_lambda = g_lambda

    def K_linear(self, X1, X2):
        return np.dot(X1, X2.T)

    def K_gaussian(self, X1, X2):
        z = X1.shape[-1]
        m, k = int(X1.size / z), int(X2.size / z)
        # norm = np.linalg.norm((X1 - X2), 2, -1)  # (k,m)
        norm = np.maximum(
            (X1**2).sum(axis=-1).reshape((1, m))
            + (X2**2).sum(axis=-1).reshape((k, 1))
            - 2 * np.dot(X1, X2.T).T,  # (k,m)
            0,
        )  # (k,m)
        prod = np.exp(-self.g_lambda * (norm))
        return prod.T

    def train(self, train_x, train_y):
        C = self.C
        print("Generating CVXOPT params")
        t = now()

        y_i, y_j = train_y.reshape((1, m)), train_y.reshape((m, 1))
        K_val = self.K(train_x, train_x)
        P = y_i * y_j * K_val
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

        # sols.dump(abs_path("data/sols_12_gaussian.npy"))

        sv_pos = sols.flatten() > 1e-6
        self.m_sv = np.count_nonzero(sv_pos)  # 3066
        self.alpha = sols[sv_pos].reshape((self.m_sv, 1))  # (m,1)
        self.X_sv = train_x[sv_pos, :]  # (m,768)
        self.y_sv = train_y[sv_pos, :]  # (m,1)
        self.b = np.mean(self.y_sv - self.wx(self.X_sv))

    def train_precomputed(self, sols):
        sv_pos = sols.flatten() > 1e-6
        self.m_sv = np.count_nonzero(sv_pos)  # 3066
        self.alpha = sols[sv_pos].reshape((self.m_sv, 1))  # (m,1)
        self.X_sv = train_x[sv_pos, :]  # (m,768)
        self.y_sv = train_y[sv_pos, :]  # (m,1)

        b = np.mean(self.y_sv - self.wx(self.X_sv))
        # p, n = self.y_sv.flatten() == 1, self.y_sv.flatten() == -1
        # b = -0.5 * np.max(self.wx(self.X_sv[n, :])) + np.min(self.wx(self.X_sv[p, :]))
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


# sols = np.load(abs_path("data/sols_12_gaussian.npy"), allow_pickle=True)
s = SVM(kernel="gaussian")
# s = SVM(kernel="linear")
s.train(train_x, train_y)
# s.train_precomputed(sols)
s.test(val_x, val_y)
