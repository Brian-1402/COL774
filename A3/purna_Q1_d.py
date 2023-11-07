import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder


label_encoder = None


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
    return X.to_numpy(), y.to_numpy()


x_train, y_train = get_np_array("train.csv")
x_test, y_test = get_np_array("test.csv")
x_val, y_val = get_np_array("val.csv")

train = []
test = []
val = []
depth = [15, 25, 35, 45]

for d in depth:
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=d)
    clf.fit(x_train, y_train)
    print(clf.get_depth())
    train.append(clf.score(x_train, y_train))
    test.append(clf.score(x_test, y_test))
    val.append(clf.score(x_val, y_val))
    print(clf.score(x_train, y_train), clf.score(x_test, y_test))

plt.plot(depth, train, label="Train", color="red")
plt.plot(depth, test, label="Test", color="blue")
# plt.plot(depth,val,label = 'Val',color = 'green')
plt.legend()
plt.title("Accuracy v/s Depth")
plt.show()

fin_depth = depth[np.argmax(val)]
print("depth:", fin_depth)
alpha = [0.001, 0.01, 0.1, 0.2]
train_alpha = []
test_alpha = []
val_alpha = []
for a in alpha:
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=100, ccp_alpha=a)
    clf.fit(x_train, y_train)
    print(clf.get_depth())
    train_alpha.append(clf.score(x_train, y_train))
    test_alpha.append(clf.score(x_test, y_test))
    val_alpha.append(clf.score(x_val, y_val))
    print(
        clf.score(x_train, y_train), clf.score(x_test, y_test), clf.score(x_val, y_val)
    )

plt.plot(alpha, train_alpha, label="Train", color="red")
plt.plot(alpha, test_alpha, label="Test", color="blue")
plt.plot(alpha, val_alpha, label="Val", color="green")
plt.legend()
plt.title("Accuracy v/s Pruning Parameter")
plt.show()

fin_ccp_alpha = alpha[np.argmax(val_alpha)]
print(fin_depth, fin_ccp_alpha)

fin_clf = DecisionTreeClassifier(
    criterion="entropy", max_depth=fin_depth, ccp_alpha=fin_ccp_alpha
)
clf.fit(x_train, y_train)
print(clf.score(x_train, y_train))
print(clf.score(x_test, y_test))
print(clf.score(x_val, y_val))

# ------------ comparison to decision trees built in (a) and (b) -------------- #
