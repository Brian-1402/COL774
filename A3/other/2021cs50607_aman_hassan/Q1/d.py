from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time

label_encoder = None 

def get_np_array(file_name):
    global label_encoder
    data = pd.read_csv(file_name)
    
    need_label_encoding = ['team','host','opp','month', 'day_match']
    if(label_encoder is None):
        label_encoder = OneHotEncoder(sparse_output = False)
        label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
    
    #merge the two dataframes
    dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    return X.to_numpy(), y.to_numpy()

def plot_graph(X_train, y_train, X_test, y_test, X_val, y_val,param: str, param_range: list,best_depth=0):
    best_param = None
    if param == "max_depth":
    # Plotting the graph
        train_acc = []
        test_acc = []
        val_acc = []
        best_depth = 0
        for i in param_range:
            dt = DecisionTreeClassifier(max_depth=i,criterion="entropy")
            dt.fit(X_train, y_train)
            train_acc.append(100*dt.score(X_train, y_train))
            test_acc.append(100*dt.score(X_test, y_test))
            val_score = 100*dt.score(X_val, y_val)
            if val_acc != []:
                if val_score > val_acc[-1]:
                    best_depth = i
            else:
                best_depth = i
            val_acc.append(val_score)
        for i in range(len(train_acc)):
            print(f"Training Accuracy for max_depth = {param_range[i]} is {train_acc[i]:.4f}")
        for i in range(len(test_acc)):
            print(f"Test Accuracy for max_depth = {param_range[i]} is {test_acc[i]:.4f}")
        best_param = best_depth
        plt.plot(param_range, train_acc, label="Training Accuracy")
        plt.plot(param_range, test_acc, label="Test Accuracy")
        plt.plot(param_range, val_acc, label="Validation Accuracy")
        plt.xlabel(param)
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
    else:
        train_acc = []
        test_acc = []
        val_acc = []
        best_ccp_alpha= 0
        for i in param_range:
            dt = DecisionTreeClassifier(max_depth=best_depth,ccp_alpha=i,criterion="entropy")
            dt.fit(X_train, y_train)
            train_acc.append(100*dt.score(X_train, y_train))
            test_acc.append(100*dt.score(X_test, y_test))
            val_score = 100*dt.score(X_val, y_val)
            if val_acc != []:
                if val_score > val_acc[-1]:
                    best_ccp_alpha = i
            else:
                best_ccp_alpha = i
            val_acc.append(val_score)
        for i in range(len(train_acc)):
            print(f"Training Accuracy for ccp_alpha = {param_range[i]} is {train_acc[i]:.4f}")
        for i in range(len(test_acc)):
            print(f"Test Accuracy for ccp_alpha = {param_range[i]} is {test_acc[i]:.4f}")
        best_param = best_ccp_alpha
        plt.plot(param_range, train_acc, label="Training Accuracy")
        plt.plot(param_range, test_acc, label="Test Accuracy")
        plt.plot(param_range, val_acc, label="Validation Accuracy")
        plt.xlabel(param)
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
    return best_param

if __name__ == '__main__':
    
    dirname = os.path.dirname(__file__)

    X_train,y_train = get_np_array(os.path.join(dirname,'../Data/a3_data_starter_code/train.csv'))
    X_test, y_test = get_np_array(os.path.join(dirname,"../Data/a3_data_starter_code/test.csv"))
    X_val, y_val = get_np_array(os.path.join(dirname,"../Data/a3_data_starter_code/val.csv"))

    best_depth = plot_graph(X_train, y_train, X_test, y_test, X_val, y_val, "max_depth", [15,25,35,45])
    best_ccp = plot_graph(X_train, y_train, X_test, y_test, X_val, y_val, "ccp_alpha", [0.001,0.01,0.1,0.2],best_depth)
    print("Best depth:", best_depth)
    print("Best ccp_alpha:", best_ccp)

'''
Output:
Max depth:
    Training Accuracy for max_depth = 15 is 71.3428
    Training Accuracy for max_depth = 25 is 85.4734
    Training Accuracy for max_depth = 35 is 94.3529
    Training Accuracy for max_depth = 45 is 99.5528

    Test Accuracy for max_depth = 15 is 60.5998
    Test Accuracy for max_depth = 25 is 63.3919
    Test Accuracy for max_depth = 35 is 64.6329
    Test Accuracy for max_depth = 45 is 64.0124
    
CCP_Alpha:
    Training Accuracy for ccp_alpha = 0.001 is 68.9408
    Training Accuracy for ccp_alpha = 0.01 is 53.4432
    Training Accuracy for ccp_alpha = 0.1 is 50.3386
    Training Accuracy for ccp_alpha = 0.2 is 50.3386

    Test Accuracy for ccp_alpha = 0.001 is 66.2875
    Test Accuracy for ccp_alpha = 0.01 is 51.8097
    Test Accuracy for ccp_alpha = 0.1 is 49.6381
    Test Accuracy for ccp_alpha = 0.2 is 49.6381

Best depth: 45
Best ccp_alpha: 0.001
'''