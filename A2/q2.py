import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cvxopt import matrix, solvers
from datetime import datetime
from sklearn.svm import SVC
from itertools import combinations

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
    def __init__(self, model="CVXOPT", kernel="linear", C=1, g=1e-3):
        self.C = C
        model_map = {
            "CVXOPT": (self.train_CVXOPT, self.predict_CVXOPT),
            "sklearn": (self.train_sklearn, self.predict_sklearn),
        }
        kernel_map = {"linear": self.K_linear, "rbf": self.K_gaussian}
        self.kernel = kernel
        self.model = model
        self.K = kernel_map[kernel]
        self.train, self.predict = model_map[model]
        self.g = g

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
        prod = np.exp(-self.g * (norm))
        return prod.T

    def train_CVXOPT(self, train_x, train_y):
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

        # sols.dump(abs_path("data/sols_12_linear.npy"))

        self.sv_pos = sv_pos = np.where(sols.flatten() > 1e-6)[0]
        # position indexes of the values of alpha !=0, indexes of support vectors
        self.m_sv = sv_pos.size  # no. of SVs : m
        self.alpha = sols[sv_pos].reshape((self.m_sv, 1))  # (m,1), alpha of SVs
        self.X_sv = train_x[sv_pos, :]  # (m,768) X values of SV
        self.y_sv = train_y[sv_pos, :]  # (m,1) y values of SV
        self.b = np.mean(self.y_sv - self.wx(self.X_sv))  # computed b
        if self.kernel == "linear":
            self.w = (self.alpha * self.y_sv * self.X_sv).sum(axis=0)

    def train_precomputed(self, sols):
        sv_pos = self.sv_pos = np.where(sols.flatten() > 1e-6)[0]
        # print(self.sv_pos)

        self.m_sv = sv_pos.size  # 3066
        self.alpha = sols[sv_pos].reshape((self.m_sv, 1))  # (m,1)
        self.X_sv = train_x[sv_pos, :]  # (m,768)
        self.y_sv = train_y[sv_pos, :]  # (m,1)

        b = np.mean(self.y_sv - self.wx(self.X_sv))
        # p, n = self.y_sv.flatten() == 1, self.y_sv.flatten() == -1
        # b = -0.5 * np.max(self.wx(self.X_sv[n, :])) + np.min(self.wx(self.X_sv[p, :]))
        self.b = b
        if self.kernel == "linear":
            self.w = (self.alpha * self.y_sv * self.X_sv).sum(axis=0)

    def wx(self, X_test):
        # X - (m,n), X_test - (k,n)
        # a,y - (1,m)
        w_x = (self.alpha * self.y_sv * self.K(self.X_sv, X_test)).sum(axis=0)
        return w_x.reshape((w_x.size, 1))

    def predict_CVXOPT(self, X_test):
        return (2.0 * (self.wx(X_test) + self.b > 0) - 1).ravel()

    def train_sklearn(self, train_x, train_y):
        sk_model = SVC(C=self.C, gamma=self.g, kernel=self.kernel, tol=1e-6)
        t = now()
        sk_model.fit(train_x, train_y.ravel())
        print("SVC training time:", now() - t)
        self.sk_model = sk_model

        self.sv_pos = sk_model.support_
        # print(self.sv_pos)
        self.m_sv = np.sum(sk_model.n_support_)
        self.alpha = np.abs(sk_model.dual_coef_).T
        self.X_sv = train_x[self.sv_pos, :]
        self.y_sv = train_y[self.sv_pos, :]
        # b = np.mean(self.y_sv - self.wx(self.X_sv))
        b = sk_model.intercept_
        self.b = b
        if self.kernel == "linear":
            self.w = (self.alpha * self.y_sv * self.X_sv).sum(axis=0)

    def predict_sklearn(self, X_test):
        return self.sk_model.predict(X_test)

    def test(self, X_test, y_test):
        ans = self.predict(X_test) == y_test.ravel()
        print(f"Accuracy: {100 * np.mean(ans) :.2f}%")

    def get_top_6(self):
        top_6 = self.alpha.flatten().argsort()[-6:]
        for i in range(6):
            w = self.X_sv[top_6[i], :] * 255
            # print(self.alpha[top_6[i]], w.shape)
            w_arr = w.reshape((16, 16, 3)).astype(np.uint8)
            # print(w_arr, w_arr.shape)
            w_img = Image.fromarray(w_arr)
            img = w_img.resize((320, 320), resample=Image.NEAREST)

            w_img.save(
                abs_path(
                    f"submission\\q2\\plots\\{self.model}_{self.kernel}_top{6-i}.png"
                )
            )

    def get_w_img(self):
        w = self.w * 255
        w_arr = w.reshape((16, 16, 3)).astype(np.uint8)
        w_img = Image.fromarray(w_arr)
        img = w_img.resize((320, 320), resample=Image.NEAREST)

        w_img.save(abs_path(f"submission\\q2\\plots\\{self.model}_w_img.png"))


sols_linear = np.load(abs_path("data/sols_34_linear.npy"), allow_pickle=True)
sols_gaussian = np.load(abs_path("data/sols_34_gaussian.npy"), allow_pickle=True)
s_l = SVM("CVXOPT", "linear")
# s_l.train(train_x, train_y)
s_l.train_precomputed(sols_linear)
s_g = SVM("CVXOPT", "rbf")
# s_g.train(train_x, train_y)
s_g.train_precomputed(sols_gaussian)


def part_a():
    print("\nPart a\nFor linear CVXOPT SVM")
    s = SVM("CVXOPT", "linear")
    # s.train(train_x, train_y)
    s.train_precomputed(sols_linear)
    print(
        f"{s.m_sv} out of {train_y.size} training examples are support vectors. {s.m_sv/train_y.size*100 :.2f} %"
    )
    s.test(val_x, val_y)
    s.get_top_6()
    s.get_w_img()


def part_b():
    print("\nPart b\nFor gaussian CVXOPT SVM")
    s = SVM("CVXOPT", "rbf")
    # s.train(train_x, train_y)
    s.train_precomputed(sols_gaussian)
    print(
        f"{s.m_sv} out of {train_y.size} training examples are support vectors. {s.m_sv/train_y.size*100 :.2f} %"
    )
    s.test(val_x, val_y)
    s.get_top_6()
    # s.get_w_img()
    print(
        "matching support vectors with linear",
        len(set(s.sv_pos).intersection(set(s_l.sv_pos))),
    )


def part_c():
    print("\nPart c\nFor linear sklearn SVM")
    sl = SVM("sklearn", "linear")
    sl.train(train_x, train_y)
    sl.test(val_x, val_y)
    model = sl.sk_model
    print(
        f"{sl.m_sv} out of {train_y.size} training examples are support vectors. \n{sl.m_sv/train_y.size*100 :.2f} %"
    )
    print(
        "matching support vectors with linear cvxopt",
        len(set(sl.sv_pos).intersection(set(s_l.sv_pos))),
    )
    print(f"for sklearn, b:{sl.b}")
    print(f"for cvxopt, b:{s_l.b}")
    print(f"norm of difference of weight values: {np.linalg.norm(sl.w-s_l.w)}")

    print("\nFor gaussian sklearn SVM")
    sg = SVM("sklearn", "rbf")
    sg.train(train_x, train_y)
    sg.test(val_x, val_y)
    print(
        f"{sg.m_sv} out of {train_y.size} training examples are support vectors. \n{sg.m_sv/train_y.size*100 :.2f} %"
    )
    print(
        "matching support vectors with gaussian cvxopt",
        len(set(sg.sv_pos).intersection(set(s_g.sv_pos))),
    )

    print(
        "matching SVs with linear and gaussian sklearn",
        len(set(sg.sv_pos).intersection(set(sl.sv_pos))),
    )


part_c()


"""Prepare multiclass data"""
train_full = []
test_full = []
for i in range(6):
    # print(f"Adding class {i}")
    train_vals = []
    for f in os.scandir(abs_path(f"data/svm/train/{i}")):
        img = Image.open(f.path)
        train_vals.append(process_img(img))
    train_full.append(np.stack(train_vals))

    test_vals = []
    for f in os.scandir(abs_path(f"data/svm/val/{i}")):
        img = Image.open(f.path)
        test_vals.append(process_img(img))
    test_full.append(np.stack(test_vals))

train_full = np.stack(train_full)
test_full = np.stack(test_full)
# print(X_values)


class SVM_Multiclass:
    def __init__(self, kernel="linear", C=1, g=1e-3):
        self.C = C
        kernel_map = {"linear": self.K_linear, "rbf": self.K_gaussian}
        self.kernel = kernel
        self.K = kernel_map[kernel]
        self.g = g
        # choices = list(combinations(classes, 2))

    def load_data(self, data):
        """data: (c,m,z) : (6,2380,768)"""
        choices = list(combinations(range(data.shape[0]), 2))
        dataset = []
        class_mappings = []
        for c in range(len(choices)):
            p, n = data[c[0]], data[c[1]]
            x = np.concatenate(p, n)
            y = np.concatenate(
                [np.ones((p.shape[0], 1)), -1 * np.ones((n.shape[0], 1))]
            )
            dataset.append(x, y)
            class_mappings.append({1: c[0], -1: c[1]})
        self.class_mappings = class_mappings
        return dataset

    def train(self, data):
        dataset = self.load_data(data)
        SVMs = []
        tx = now()
        for i in range(len(dataset)):
            s = SVM("CVXOPT", "rbf")
            t = now()
            print(f"training {i}")
            s.train(dataset[i][0], dataset[i][1])
            SVMs.append(s)
            print(f"time taken for {i}:", now() - t)
        print(f"total time taken", now() - tx)

    def predict(self, X_test):
        pass
