import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

label_encoder = None


def extract_data(file):
    data = np.transpose(pd.read_csv(file, header=None).to_numpy()[1:])
    y = data[(len(data) - 1)]
    x = np.array(data[1 : (len(data) - 1)])
    return x, y


def get_np_array(file_name):
    global label_encoder
    data = pd.read_csv(file_name)

    need_label_encoding = ["team", "host", "opp", "month", "day_match"]
    if label_encoder is None:
        label_encoder = OneHotEncoder(sparse_output=False)
        label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(
        label_encoder.transform(data[need_label_encoding]),
        columns=label_encoder.get_feature_names_out(),
    )

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

    X = final_data.iloc[:, :-1]
    y = final_data.iloc[:, -1:]
    return X.to_numpy(), y.to_numpy().flatten()


x_train, y_train = get_np_array("train.csv")
x_test, y_test = get_np_array("test.csv")
x_val, y_val = get_np_array("val.csv")


clf = RandomForestClassifier(
    max_features=0.7, min_samples_split=8, n_estimators=350, oob_score=True
)
clf.fit(x_train, y_train)
print("oob: ", clf.oob_score_)
print("train data: ", clf.score(x_train, y_train))
print("test data: ", clf.score(x_test, y_test))
print("val data: ", clf.score(x_val, y_val))


param_grid = {
    "n_estimators": [50, 150, 250, 350],
    "max_features": [0.1, 0.3, 0.5, 0.7, 0.9],
    "min_samples_split": [2, 4, 6, 8, 10],
}
gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, verbose=4)
gs.fit(x_train, y_train)
print("train score:", gs.score(x_train, y_train))
print("test score:", gs.score(x_test, y_test))
print("val score:", gs.score(x_val, y_val))
print(gs.best_estimator_)
