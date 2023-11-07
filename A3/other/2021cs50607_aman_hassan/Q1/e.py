from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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


def grid_search(X_train,y_train,X_test,y_test,X_val,y_val,param: dict):
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    y_val = y_val.ravel()  
    Forest = RandomForestClassifier(criterion="entropy",oob_score=True)
    grid = GridSearchCV(Forest, param,verbose=4)
    grid.fit(X_train,y_train)
    best_param = grid.best_params_
    # best_param = {'max_features': 0.7, 'min_samples_split': 8, 'n_estimators': 150}
    Forest = RandomForestClassifier(criterion="entropy",oob_score=True,**best_param)
    Forest.fit(X_train,y_train)
    print(f"Best_param = {best_param}")
    print(f"Out of Box Accuracy: {100*Forest.oob_score_:.4f}")
    print(f"Training Accuracy: {100*Forest.score(X_train,y_train):.4f}")
    print(f"Test Accuracy: {100*Forest.score(X_test,y_test):.4f}")
    print(f"Validation Accuracy: {100*Forest.score(X_val,y_val):.4f}")

if __name__ == '__main__':
    
    dirname = os.path.dirname(__file__)

    X_train,y_train = get_np_array(os.path.join(dirname,'../Data/a3_data_starter_code/train.csv'))
    X_test, y_test = get_np_array(os.path.join(dirname,"../Data/a3_data_starter_code/test.csv"))
    X_val, y_val = get_np_array(os.path.join(dirname,"../Data/a3_data_starter_code/val.csv"))

    params = {'n_estimators': [50,150,250,350], 'max_features': [0.1,0.3,0.5,0.7,0.9],"min_samples_split": [2,4,6,8,10]}
    grid_search(X_train,y_train,X_test,y_test,X_val,y_val,params)

'''
Output:
Best_param = {'max_features': 0.7, 'min_samples_split': 8, 'n_estimators': 150}
Out of Box Accuracy: 71.8922
Training Accuracy: 98.7990
Test Accuracy: 71.7684
Validation Accuracy: 69.5402
'''