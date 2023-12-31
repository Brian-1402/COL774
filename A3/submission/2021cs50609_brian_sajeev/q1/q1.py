from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import os
import copy
import time

import matplotlib.pyplot as plt

label_encoder = None


def abs_path(relative_pos):
    return os.path.join(os.path.dirname(__file__), relative_pos)


def get_np_array(file_name, one_hot=False):
    global label_encoder
    data = pd.read_csv(file_name)
    need_label_encoding = ["team", "host", "opp", "month", "day_match"]
    if label_encoder is None:
        if one_hot:
            label_encoder = OneHotEncoder(sparse_output=False)
        else:
            label_encoder = OrdinalEncoder()

        label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(
        label_encoder.transform(data[need_label_encoding]),
        columns=label_encoder.get_feature_names_out(),
    )
    a = label_encoder.categories_
    # merge the two dataframes
    dont_need_label_encoding = [
        "year",
        "toss",
        "bat_first",
        "format",
        "fow",
        "score",
        "rpo",
        "result",
    ]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    labels = list(final_data.columns)[:-1]
    features_info = dict()
    features_info["num_features"] = len(labels)
    features_info["names"] = labels
    features_info["values"] = [[0, 1] for i in range(len(labels))]
    if not one_hot:
        for i in range(len(label_encoder.categories_)):  # 0 to 4
            features_info["values"][i] = list(range(len(label_encoder.categories_[i])))
    features_info["values"][-7] = None  # year
    features_info["values"][-3] = None  # fow
    features_info["values"][-2] = None  # score
    features_info["values"][-1] = None  # rpo

    features_info["types"] = [
        "cat",  # 0. team
        "cat",  # 1. host
        "cat",  # 2. opp
        "cat",  # 3. month
        "cat",  # 4. day_match
        "cont",  # 5. year
        "cat",  # 6. toss
        "cat",  # 7. bat_first
        "cat",  # 8. format
        "cont",  # 9. fow
        "cont",  # 10. score
        "cont",  # 11. rpo
    ]
    # features_info["value_names"]=[]
    if one_hot:
        while len(features_info["types"]) != len(labels):
            features_info["types"] = ["cat"] + features_info["types"]
        # features_info["values"] = features_info["values"][5:]
        # while len(features_info["values"]) != len(labels):
        #     features_info["values"] = [0, 1] + features_info["values"]

    X = final_data.iloc[:, :-1]
    y = final_data.iloc[:, -1:]

    return X.to_numpy(), y.to_numpy(), features_info


class DTNode:
    def __init__(self, depth, is_leaf=False, value=0, column=None, median=None):
        # to split on column
        self.depth = depth

        # add children afterwards
        self.children = None

        self.is_leaf = is_leaf
        # if leaf then also need value
        # if self.is_leaf:
        #     self.value = value
        self.value = value

        if not self.is_leaf:
            self.column = column
            # column here means column to split children on

        # if Node column is continuous, median for splitting:
        self.median = median

        self.descendants = 0

    def infer_child(self, X):
        """
        Args:
            X: A single example np array [num_features]
        Returns:
            child: A DTNode
        """
        return self.children[X[self.column]]

    def add_child(self, child):
        if self.children is None:
            self.children = []
        self.children.append(child)


class DTTree:
    def __init__(self):
        # Tree root should be DTNode
        self.root = None
        self.features_info = None

    def prob(self, y):
        """
        Calculate probability of y=1
        O(n)
        """

        return 0 if len(y) == 0 else np.sum(y) / len(y)

    def calc_entropy(self, y):
        """
        Calculates entropy of y
        Args:
            y: numpy array of shape [num_samples, 1]
        Returns:
            entropy: scalar value
        """
        prob = self.prob(y)
        if prob == 0 or prob == 1:
            return 0
        return -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)

    def get_split_conditions(self, X, feature, features_info):
        if features_info["types"][feature] == "cat":
            split_conditions = [
                X[:, feature] == j for j in features_info["values"][feature]
            ]
            pass
        elif features_info["types"][feature] == "cont":
            # if continuous, then split on median of subset of X given
            median = np.median(X[:, feature])
            split_conditions = [X[:, feature] <= median, X[:, feature] > median]
        return split_conditions

    def choose_best_feature(self, X, y, features_used, features_info):
        """
        Args:
            X: numpy array of data [num_samples, num_features]
            y: numpy array of classes [num_samples, 1]
            features_used: list of indices of features to consider for splitting
            features_info: dict containing info about features
        Returns:
            best_feature: index of best feature to split on
            split_conditions: list of split conditions for best feature.
                eg: if best feature is 2 and it is continious then
                    split_conditions = [X[:,2] <= 4.5, X[:,2] > 4.5]
                or if best feature is 3 and it is categorical then (unique values = 2,3,4)
                    split_conditions = [X[:,3] == 2, X[:,3] == 3, X[:,3] == 4]
        """
        mutual_info = np.zeros(features_info["num_features"])
        H_y = self.calc_entropy(y)
        for i in range(features_info["num_features"]):
            conditions = self.get_split_conditions(X, i, features_info)
            # if features_info["types"][i] == "cat" and i in features_used:
            #     continue
            # if features_info["types"][i] == "cat" and y[X[:, i] == ].shape[0] == 0:
            #     continue
            mutual_info[i] = H_y

            for condition in conditions:
                # print(np.all(condition))
                if np.all(condition) == True:
                    mutual_info[i] = -1
                    break
                mutual_info[i] -= (
                    self.calc_entropy(y[condition]) * len(y[condition]) / len(y)
                )
        best = np.argmax(mutual_info)
        return best, self.get_split_conditions(X, best, features_info)

    def fit_recur(self, X, y, features_used, features_info, max_depth, depth):
        # if depth == max_depth:
        #     return DTNode(depth, True, value=self.prob(y))
        if len(y) == 0:
            return None
        p_y = self.prob(y)
        # print(depth * "|  " + f"depth: {depth}", end="")
        if (
            p_y == 0.0
            or p_y == 1.0
            or len(features_used) == features_info["num_features"]
            or depth == max_depth
        ):
            # print(f", is leaf, p: {p_y}\n")
            return DTNode(depth, True, value=p_y)

        feature, split_conditions = self.choose_best_feature(
            X, y, features_used, features_info
        )
        features_used.append(feature)
        median = (
            np.median(X[:, feature])
            if features_info["types"][feature] == "cont"
            else None
        )
        node = DTNode(depth, False, value=p_y, column=feature, median=median)
        loop = 0
        # print(f", feature: {feature}, is NOT leaf, entering loop")
        for condition in split_conditions:
            # print(depth * " " + f"Loop: {loop}")
            loop += 1
            child = self.fit_recur(
                X[condition],
                y[condition],
                copy.deepcopy(features_used),
                features_info,
                max_depth,
                depth + 1,
            )
            node.add_child(child)
            node.descendants += (1 + child.descendants) if child is not None else 0
        # print()
        return node

    def fit(self, X, y, features_info, max_depth=10):
        """
        Makes decision tree
        Args:
            X: numpy array of data [num_samples, num_features]
            y: numpy array of classes [num_samples, 1]
            types: list of [num_features] with types as: cat, cont
                eg: if num_features = 4, and last 2 features are continious then
                    types = ['cat','cat','cont','cont']
            max_depth: maximum depth of tree
        Returns:
            None
        """
        self.features_info = features_info

        self.root = self.fit_recur(
            X,
            y,
            features_used=list(),
            features_info=features_info,
            max_depth=max_depth,
            depth=0,
        )

    def inference(self, x):
        """
        Runs inference for a single test example.
        """
        node = self.root
        while not node.is_leaf:
            children = node.children
            if self.features_info["types"][node.column] == "cat":
                values = self.features_info["values"][node.column]
                next_node = children[values[int(x[node.column])]]
            elif self.features_info["types"][node.column] == "cont":
                if x[node.column] <= node.median:
                    next_node = children[0]
                else:
                    next_node = children[1]
            if next_node is None:
                return node.value
            node = next_node
        return node.value

    def __call__(self, X):
        """
        Predicted classes for X
        Args:
            X: numpy array of data [num_samples, num_features]
        Returns:
            y: [num_samples, 1] predicted classes
        """
        # inf = np.vectorize(self.inference)
        y = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            p = self.inference(X[i])
            y[i] = 1 if p > 0.5 else 0
        return y

    def prune_recur(self, node, data, base_acc, plt_data):
        if node.is_leaf:
            return
        else:
            for child in node.children:
                self.prune_recur(child, data, base_acc, plt_data)
            node.is_leaf = True

            X_train, y_train, X_test, y_test, X_val, y_val, features_info = data
            train_acc = np.sum(self(X_train) == y_train) / len(y_train) * 100
            test_acc = np.sum(self(X_test) == y_test) / len(y_test) * 100
            val_acc = np.sum(self(X_val) == y_val) / len(y_val) * 100

            if val_acc > base_acc[0]:
                # print(f"Pruned at depth {node.depth}, new accuracy: {val_acc:.4f}")
                base_acc[0] = val_acc
                node.children = None
                self.root.descendants -= node.descendants + 1

                plt_data["desc"].append(self.root.descendants)
                plt_data["val_acc"].append(val_acc)
                plt_data["test_acc"].append(test_acc)
                plt_data["train_acc"].append(train_acc)
            else:
                # print(
                #     f"not pruned, depth {node.depth}, {base_acc[0]:.2f}, {val_acc:.4f}"
                # )
                node.is_leaf = False

    def post_prune(self, data):
        plt_data = dict()
        plt_data["desc"] = []
        plt_data["val_acc"] = []
        plt_data["test_acc"] = []
        plt_data["train_acc"] = []
        val_acc = np.sum(self(data[-3]) == data[-2]) / len(data[-2]) * 100
        self.prune_recur(self.root, data, [val_acc], plt_data)
        return plt_data


def get_data(one_hot=False):
    global label_encoder
    label_encoder = None
    X_train, y_train, features_info = get_np_array(
        abs_path("data/q1/train.csv"), one_hot
    )
    X_test, y_test, _ = get_np_array(abs_path("data/q1/test.csv"), one_hot)
    X_val, y_val, _ = get_np_array(abs_path("data/q1/val.csv"), one_hot)
    return X_train, y_train, X_test, y_test, X_val, y_val, features_info


def _part_a_b(one_hot=False, prune=False, depths=[5, 10, 15, 20, 25]):
    # change the path if you want
    X_train, y_train, X_test, y_test, X_val, y_val, features_info = get_data(one_hot)
    train_accs = []
    test_accs = []
    for depth in depths:
        tree = DTTree()
        t = time.time()
        tree.fit(X_train, y_train, features_info, max_depth=depth)
        t2 = time.time() - t
        train_acc = np.sum(tree(X_train) == y_train) / len(y_train) * 100
        test_acc = np.sum(tree(X_test) == y_test) / len(y_test) * 100
        print(
            f"Time: {t2:.2f}s, Depth: {depth}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%",
        )
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    return train_accs, test_accs


def part_a():
    import matplotlib.pyplot as plt

    print("\nPart A:")
    depths = [5, 10, 15, 20, 25]
    train_accs, test_accs = _part_a_b(one_hot=False, prune=False, depths=depths)
    plt.plot(depths, train_accs, label="Train Acc", color="red", marker="o")
    plt.plot(depths, test_accs, label="Test Acc", color="blue", marker="o")
    plt.legend(loc="best")
    plt.xlabel("Depth")
    plt.ylabel("Accuracies")
    plt.title("Part A")
    plt.axis([None, None, 0, 105])

    # plt.show()
    plt.savefig(abs_path(f"./report/images/q1_part_a.png"))

    plt.clf()


def part_b():
    import matplotlib.pyplot as plt

    print("\nPart B:")
    depths = [15, 25, 35, 45]
    print("Using one-hot encoding:")
    train_accs, test_accs = _part_a_b(one_hot=True, prune=False, depths=depths)
    # fig, plt = plt.subplots()
    plt.plot(depths, train_accs, label="Train Acc one-hot", color="red", marker="o")
    plt.plot(depths, test_accs, label="Test Acc one-hot", color="blue", marker="o")
    # plt.legend(loc="best")
    # plt.xlabel("Depth")
    # plt.ylabel("Accuracies")
    # plt.title("With One-hot encoding")
    # plt.savefig(abs_path(f"./report/images/q1_part_b.png"))

    # plt.set(ylim=[50, 110])
    # plt.show()
    print("\nWithout one-hot encoding:")
    a_train_accs, a_test_accs = _part_a_b(one_hot=False, prune=False, depths=depths)
    plt.plot(depths, a_train_accs, label="Train Acc k-way", color="orange", marker="o")
    plt.plot(depths, a_test_accs, label="Test Acc k-way", color="teal", marker="o")
    plt.legend(loc="best")
    plt.xlabel("Depth")
    plt.ylabel("Accuracies")
    plt.title("Part B")
    plt.axis([None, None, 0, 105])

    # plt.set(ylim=[50, 110])
    plt.savefig(abs_path(f"./report/images/q1_part_b.png"))
    # plt.show()
    plt.clf()


def part_c():
    print("\nPart C:")
    depths = [15, 25, 35, 45]
    print("Pruning, with one-hot encoding:")
    X_train, y_train, X_test, y_test, X_val, y_val, features_info = get_data(True)
    for depth in depths:
        tree = DTTree()
        t = time.time()
        tree.fit(X_train, y_train, features_info, max_depth=depth)
        plt_data = tree.post_prune(
            (X_train, y_train, X_test, y_test, X_val, y_val, features_info)
        )

        t2 = time.time() - t
        train_acc = np.sum(tree(X_train) == y_train) / len(y_train) * 100
        test_acc = np.sum(tree(X_test) == y_test) / len(y_test) * 100
        val_acc = np.sum(tree(X_val) == y_val) / len(y_val) * 100
        print(
            f"Time: {t2:.2f}s, Depth: {depth}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Val Acc: {val_acc:.2f}%",
        )
        desc, train_accs, test_accs, val_accs = (
            plt_data["desc"],
            plt_data["train_acc"],
            plt_data["test_acc"],
            plt_data["val_acc"],
        )

        plt.plot(desc, train_accs, label="Train Acc", color="red")
        plt.plot(desc, test_accs, label="Test Acc", color="blue")
        plt.plot(desc, val_accs, label="Validation Acc", color="orange")
        plt.legend(loc="best")
        plt.xlabel("Number of existing nodes")
        plt.ylabel("Accuracies")
        plt.title(f"Part C, Depth: {depth}")
        plt.axis([max(desc), min(desc), 0, 100])
        plt.savefig(abs_path(f"./report/images/q1_part_c_depth{depth}.png"))
        # plt.show()
        plt.clf()


from sklearn.tree import DecisionTreeClassifier


def part_d():
    depths = [15, 25, 35, 45]
    print("Part D")
    X_train, y_train, X_test, y_test, X_val, y_val, features_info = get_data(True)
    train_accs, test_accs, val_accs = [], [], []
    for depth in depths:
        tree = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        t = time.time()
        tree.fit(X_train, y_train)
        t2 = time.time() - t

        train_accs.append(tree.score(X_train, y_train) * 100)
        test_accs.append(tree.score(X_test, y_test) * 100)
        val_accs.append(tree.score(X_val, y_val) * 100)
        print(
            f"Time: {t2:.2f}s, Depth: {depth}, Train Acc: {train_accs[-1]:.2f}%, Test Acc: {test_accs[-1]:.2f}%, Val Acc: {val_accs[-1]:.2f}%",
        )
    plt.plot(depths, train_accs, label="Train", color="red", marker="o")
    plt.plot(depths, test_accs, label="Test", color="blue", marker="o")
    plt.plot(depths, val_accs, label="Val", color="orange", marker="o")
    plt.legend(loc="best")
    plt.xlabel("Depth")
    plt.ylabel("Accuracies")
    plt.axis([None, None, 0, 100])

    plt.savefig(abs_path(f"./report/images/q1_part_d_depths.png"))
    # plt.show()
    plt.clf()

    train_accs, test_accs, val_accs = [], [], []
    alphas = [0.001, 0.01, 0.1, 0.2]
    for alpha in alphas:
        tree = DecisionTreeClassifier(criterion="entropy", ccp_alpha=alpha)
        t = time.time()
        tree.fit(X_train, y_train)
        t2 = time.time() - t

        print(tree.get_depth())
        train_accs.append(tree.score(X_train, y_train) * 100)
        test_accs.append(tree.score(X_test, y_test) * 100)
        val_accs.append(tree.score(X_val, y_val) * 100)
        print(
            f"Time: {t2:.2f}s, Depth: {depth}, Train Acc: {train_accs[-1]:.2f}%, Test Acc: {test_accs[-1]:.2f}%, Val Acc: {val_accs[-1]:.2f}%",
        )
    plt.plot(alphas, train_accs, label="Train", color="red", marker="o")
    plt.plot(alphas, test_accs, label="Test", color="blue", marker="o")
    plt.plot(alphas, val_accs, label="Val", color="orange", marker="o")
    plt.legend(loc="best")
    plt.xlabel("ccp_alpha")
    plt.ylabel("Accuracies")
    plt.axis([None, None, 0, 100])

    plt.savefig(abs_path(f"./report/images/q1_part_d_pruning.png"))
    # plt.show()
    plt.clf()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def part_e():
    X_train, y_train, X_test, y_test, X_val, y_val, features_info = get_data(True)
    y_train = y_train.flatten()
    tree = RandomForestClassifier(criterion="entropy", oob_score=True)
    tree.fit(X_train, y_train)
    print("oob: ", tree.oob_score_)
    print("train data: ", tree.score(X_train, y_train) * 100)
    print("test data: ", tree.score(X_test, y_test) * 100)
    print("val data: ", tree.score(X_val, y_val) * 100)

    param_grid = {
        "n_estimators": [50, 150, 250, 350],
        "max_features": [0.1, 0.3, 0.5, 0.7, 0.9],
        "min_samples_split": [2, 4, 6, 8, 10],
    }
    grid = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5, verbose=4)
    grid.fit(X_train, y_train)
    print("train score:", grid.score(X_train, y_train) * 100)
    print("test score:", grid.score(X_test, y_test) * 100)
    print("val score:", grid.score(X_val, y_val) * 100)
    print(grid.best_estimator_)


if __name__ == "__main__":
    part_b()
