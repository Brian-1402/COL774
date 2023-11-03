from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import os
import copy

label_encoder = None


def abs_path(relative_pos):
    return os.path.join(os.path.dirname(__file__), relative_pos)


def get_np_array(file_name):
    global label_encoder
    data = pd.read_csv(file_name)
    #! How are we sure that categorical data is in the same order in train and test
    #! Also how do we know that all the categories are present in both train and test
    #! Need to maintain a separate list of categories for each column
    need_label_encoding = ["team", "host", "opp", "month", "day_match"]
    if label_encoder is None:
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

    features_info = dict()
    features_info["num_features"] = len(final_data.columns) - 1
    features_info["names"] = (need_label_encoding + dont_need_label_encoding)[:-1]
    features_info["values"] = [None] * len(features_info["names"])
    for i in range(len(label_encoder.categories_)):  # 0 to 4
        features_info["values"][i] = list(range(len(label_encoder.categories_[i])))
    features_info["values"][6] = [0, 1]
    features_info["values"][7] = [0, 1]
    features_info["values"][8] = [0, 1]
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

    def post_prune(self, X_val, y_val):
        # TODO
        pass


if __name__ == "__main__":
    # change the path if you want
    X_train, y_train, features_info = get_np_array(abs_path("data/q1/train.csv"))
    X_test, y_test, _ = get_np_array(abs_path("data/q1/test.csv"))
    max_depth = 45
    tree = DTTree()
    tree.fit(X_train, y_train, features_info, max_depth=max_depth)
    y_test_infers = tree(X_train)
    print(f"Accuracy: { np.sum(y_test_infers == y_train)/len(y_train)*100 :.2f}%")
    pass
