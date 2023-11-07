import numpy as np
import sys
import pdb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import os
from copy import deepcopy
import time


def abs_path(relative_pos):
    return os.path.join(os.path.dirname(__file__), relative_pos)


def get_data(x_path, y_path):
    """
    Args:
        x_path: path to x file
        y_path: path to y file
    Returns:
        x: np array of [NUM_OF_SAMPLES x n]
        y: np array of [NUM_OF_SAMPLES]
    """
    x = np.load(x_path)
    y = np.load(y_path)

    y = y.astype("float")
    x = x.astype("float")

    # normalize x:
    x = 2 * (0.5 - x / 255)
    return x, y


def get_metric(y_true, y_pred):
    """
    Args:
        y_true: np array of [NUM_SAMPLES x r] (one hot)
                or np array of [NUM_SAMPLES]
        y_pred: np array of [NUM_SAMPLES x r] (one hot)
                or np array of [NUM_SAMPLES]

    """
    results = classification_report(y_pred, y_true)
    print(results)


class NeuralLayer:
    """
    To help visualize dimensions, inputs from previous layer are horizontal,
    outputs are vertical, and values corresponding to perceptrons, like delj, are vertical.
    Like how perceptrons are placed vertically, and values for each perceptron are horizontal.
    So the weights theta have rows corresponding to inputs, and columns corresponding to perceptrons.
    """

    def __init__(
        self,
        input_size,
        output_size,
        activation,
        is_input=False,
        is_output=False,
        eta=1e-2,
    ):
        self.theta = np.random.randn(output_size, input_size) / np.sqrt(input_size)
        self.theta = np.concatenate((self.theta, np.zeros((output_size, 1))), axis=1)
        # theta shape = (output_size, input_size+1)
        self.delj = None  # shape = (m, output_size)
        self.input = None  # shape = (m, input_size+1)
        self.output = None  # shape = (m, output_size)
        self.is_input = is_input
        self.is_output = is_output
        self.activation = activation
        self.eta = eta

    def activ(self, netj):
        if self.activation == "relu":
            return np.where(netj > 0, netj, 0)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-netj))
        elif self.activation == "softmax":
            netj -= np.max(netj)
            exps = np.exp(netj)
            return exps / np.sum(exps, axis=1, keepdims=True)

    def del_activ(self, netj):
        if self.activation == "relu":
            return np.where(netj > 0, 1, 0.001)
        elif self.activation == "sigmoid":
            output = self.activ(netj)
            return output * (1 - output)

    def layer_output(self):
        if not self.is_output:
            output = self.output
        else:
            predicted_indices = np.argmax(self.output, axis=1)
            output = np.eye(self.output.shape[1])[predicted_indices]
        return output

    def push_forward(self, inputs):
        #! full copying over inputs is happening just for adding bias terms via concatenation. Could be avoided.
        # one alternative is to consider a "bias perceptron" for which everything gets computed normally,
        # but in the end, the output is replaced to 1. This change is made in activation funcs
        # t = time.time()
        self.input = np.concatenate((inputs, np.ones((inputs.shape[0], 1))), axis=1)
        # self.input = inputs
        self.netj = self.input @ self.theta.T  # shape = (m, output_size)
        self.output = self.activ(self.netj)
        # print(time.time() - t)
        output = self.layer_output()
        return output

    def push_backward(self, delj_down=None, theta_down=None, y=None):
        if not self.is_output:
            del_ac = self.del_activ(self.netj)
            del_ac.shape = (del_ac.shape[0], del_ac.shape[2], del_ac.shape[1])
            self.delj = (theta_down.T[:-1, :] @ delj_down) * del_ac

            input_down = np.concatenate(
                (self.output, np.ones((self.output.shape[0], 1))), axis=1
            )
            # theta_down -= (
            #     self.eta * np.sum(delj_down * input_down, axis=0) / delj_down.shape[0]
            # )
            delj_down_ = delj_down.reshape(delj_down.shape[1], delj_down.shape[0])
            # (m, output_size, 1) -> (output, m)
            input_down_ = input_down.squeeze()
            # (m, 1, input_size+1) -> (m, input+1)
            mult = delj_down_ @ input_down_
            del_theta_down = self.eta * mult / delj_down.shape[0]
            theta_down += del_theta_down
            #! Experimental, assuming pass by reference and updating theta of downstream layer
            # print(del_theta_down[0])

            # make a copy of theta before updating it
            # back_theta = deepcopy(self.theta)
            # self.theta += self.delj * self.input

        else:
            delj_ = -(self.output - y)
            delj_.shape = (delj_.shape[0], delj_.shape[2], delj_.shape[1])
            self.delj = delj_
        if self.is_input:
            # self.theta -= (
            #     self.eta * np.sum(self.delj * self.input, axis=0) / self.delj.shape[0]
            # )
            delj_ = self.delj.reshape(self.delj.shape[1], self.delj.shape[0])
            input_ = self.input.squeeze()
            del_theta = self.eta * (delj_ @ input_) / self.delj.shape[0]
            self.theta += del_theta
            # print(del_theta[0])

        return self.delj, self.theta, None


class NeuralNetwork:
    def __init__(self, layout, input_size, output_size, activation="relu"):
        # r - output layer size
        self.layers = []

        layout.append(output_size)

        for i in range(len(layout)):
            is_input, is_output = (i == 0), (i == len(layout) - 1)
            l = NeuralLayer(
                input_size,
                layout[i],
                "softmax" if is_output else activation,
                is_input,
                is_output,
            )
            self.layers.append(l)
            input_size = layout[i]

    def forward_prop(self, inputs):
        for layer in self.layers:
            inputs = layer.push_forward(inputs)
        return inputs  # final inference output

    def backward_prop(self, y):
        delj, theta = None, None
        for layer in reversed(self.layers):
            delj, theta, y = layer.push_backward(delj, theta, y)

    def give_acc(self, y_pred, y_test):
        y_pred = np.argmax(y_pred, axis=2)
        y_test = np.argmax(y_test, axis=2)
        return np.mean(y_pred == y_test) * 100

    def train(self, X, y, epochs, M=32):
        # M is batch size
        out = None
        m = X.shape[0]  # no. of samples
        X = X.reshape((X.shape[0], 1, X.shape[1]))  # (m,1,n)
        y = y.reshape((y.shape[0], 1, y.shape[1]))  # (m,1,r)

        for i in range(epochs):
            perm = np.random.permutation(m)
            # print(perm)
            # time.sleep(5)

            Xb, yb = X[perm], y[perm]
            print(f"Epoch {i+1}")
            c = 0
            for b in range(1, m // M + 1):
                B, y_b = Xb[(b - 1) * M : b * M, ...], yb[(b - 1) * M : b * M, ...]
                out = self.forward_prop(B)
                # print(out, "\n\n\n")
                # print(np.sum(out, axis=0), "")
                # print(np.sum(y_b, axis=0), "\n\n\n")
                # print(self.layers[-1].output[0], "\n\n\n")

                self.backward_prop(y_b)
                # print(self.layers[-1].delj[0])
                # time.sleep(0.2)
                # if c == 1:
                #     return
                # print(f"Batch {b} Accuracy: {self.give_acc(out, y_b) :.2f}%")
                c += 1

            # out = self.forward_prop(Xb)
            # print(self.layers[-1].delj)
            # time.sleep(5)
            print(f"Accuracy: {self.give_acc(out, y_b) :.4f}%")

    def __call__(self, X):
        inputs = X.reshape((X.shape[0], 1, X.shape[1]))
        output = self.forward_prop(inputs)
        output = output.reshape((output.shape[0], output.shape[2]))
        return output
        # return np.argmax(output, axis=1).reshape((output.shape[0]))


if __name__ == "__main__":
    global y_test_onehot

    x_train_path = abs_path("data/q2/x_train.npy")
    y_train_path = abs_path("data/q2/y_train.npy")

    X_train, y_train = get_data(
        x_train_path, y_train_path
    )  # shape = (m, n), n - input size, m - number of samples

    x_test_path = abs_path("data/q2/x_test.npy")
    y_test_path = abs_path("data/q2/y_test.npy")

    X_test, y_test = get_data(x_test_path, y_test_path)

    # you might need one hot encoded y in part a,b,c,d,e
    label_encoder = OneHotEncoder(sparse_output=False)
    label_encoder.fit(np.expand_dims(y_train, axis=-1))

    y_train_onehot = label_encoder.transform(np.expand_dims(y_train, axis=-1))
    y_test_onehot = label_encoder.transform(np.expand_dims(y_test, axis=-1))

    nn = NeuralNetwork([512], X_train.shape[1], 5, "relu")
    nn.train(X_train, y_train_onehot, 200, 32)
    y_pred = nn(X_test)
    print(
        f"Accuracy: {100 * ((y_pred @ y_test_onehot.T).trace() / y_test.shape[0]) :.2f}%"
    )
    # print(f"Accuracy: {np.sum(y_pred == y_test) / y_test.shape[0] * 100 :.2f}%")
    print(y_pred.shape)
    print(y_test.shape)
    pass


def part_d_I_think():
    clf = MLPClassifier(
        hidden_layer_sizes=(512),
        max_iter=200,
        activation="relu",
        solver="sgd",
        batch_size=32,
        learning_rate="invscaling",
        verbose=True,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    get_metric(y_test, y_pred)
    print("Accuracy: ", clf.score(X_test, y_test))
