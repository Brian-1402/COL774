from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time

label_encoder = None 

def get_np_array(file_name):
    global label_encoder
    data = pd.read_csv(file_name)
    
    need_label_encoding = ['team','host','opp','month', 'day_match']
    if(label_encoder is None):
        label_encoder = OrdinalEncoder()
        label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
    
    #merge the two dataframes
    dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    return X.to_numpy(), y.to_numpy()



class DTNode:

    def __init__(self, depth, is_leaf = False, value = 0, column:list = None):

        
        self.depth = depth

        #add children afterwards
        self.children = []

        #if leaf then also need value
        self.is_leaf = is_leaf
        self.value = value
        
        #the column/attribute on which to split on and the type of attribute it is
        #column[0] is the column number (aka attribute to split on)
        #column[1] is the type of attribute it is (cat or cont)
        #column[2] is the value to split on (for cont) or the possible categories (for cat)
        #if column[1] is cont then we check the input with column[2] and if it is less than or equal to column[2] then we go to self.child[0]
        #similarly if column[1] is cat then we check input with all i in column[2] and go to the child[i] (if  input == column[2][i]) 
        if(not self.is_leaf):
            self.column = column 



class DTTree:

    def __init__(self):
        #Tree root should be DTNode
        self.root = None
        self.nodes = 0

    def fit(self, X, y, types, curr_depth, max_depth = 10): #Basically grow tree
        '''
        Makes decision tree
        Args:
            X: numpy array of data [num_samples, num_features]
            y: numpy array of classes [num_samples, 1]
            types: list of [num_features] with types as: cat, cont
                eg: if num_features = 4, and last 2 features are continious then
                    types = ['cat','cat','cont','cont']
            max_depth: maximum depth of tree
        Returns:
            DTNode: root node of decision tree
        '''

        ''' 
        Workflow:
        1. Check if all y are same, if yes then make a leaf node and return
        2. Check if max_depth reached, if yes then make a leaf node and return
        3. Find the best feature to split on
        4. Make a node and split on the feature
        5. Recursively call fit on the splits
        '''
        # print(curr_depth)
        y = y.ravel()
        if (y.size==0): # Handling cases where median itself causes entire split to be empty
            self.nodes+=1
            return None
        if curr_depth==max_depth or np.all(y==y[0]):
            prediction_val = np.argmax(np.bincount(y))
            self.nodes+=1
            return DTNode(curr_depth, is_leaf = True, value = prediction_val)
        best_feature = self.find_best_feature(X,y,types) #this returns the required self.column (list) to split on as well as the X's and y's after split, and the value of mi
        if best_feature[-1] < 0: #Max mutual info is negative i.e. no feature is good enough to split on
            prediction_val = np.argmax(np.bincount(y))
            self.nodes+=1
            return DTNode(curr_depth, is_leaf = True, value = prediction_val)
        Node = DTNode(curr_depth, is_leaf = False, value = np.argmax(np.bincount(y)), column = best_feature[0])
        for i in range(len(best_feature[1])):
            child_node = self.fit(best_feature[1][i],best_feature[2][i],types,curr_depth+1,max_depth)
            if child_node == None:
                Node.children.append(DTNode(curr_depth+1, is_leaf = True, value = np.argmax(np.bincount(y))))
                # Node.is_leaf = True
                # Node.value = np.argmax(np.bincount(y))
                # Node.column = None
                # break
            else:
                Node.children.append(child_node)
        if curr_depth == 0:
            self.root = Node
        self.nodes+=1
        return Node
    
    def find_best_feature(self,X: np.array,y:np.array,types: list[str]):
        '''
        Choose attribute based of max decrease of mutual information
        Workflow:
        1. Calculate mutual information of each feature
        2. Return the feature which has maximum mutual information
        Additionaly - Since we are anyway splitting X, y to calculate mutual information we can return the splits as well
        '''
        mutual_info = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            # print(types[i])
            if types[i] == 'cat':
                mutual_info[i] = self.mutual_information_cat(X[:,i],y)
            else:
                mutual_info[i] = self.mutual_information_cont(X[:,i],y)
        # print(mutual_info)
        split_feature = np.argmax(mutual_info)
        # print(split_feature)
        if types[split_feature] == 'cat':
            categories = np.unique(X[:,split_feature])
            # print(categories)
            X_split = [None]*categories.shape[0]
            y_split = [None]*categories.shape[0]
            for i in range(categories.shape[0]):
                # print(X[X[:,split_feature]==categories[i]].shape)   
                # print(X_split.shape)
                X_split[i] = X[X[:,split_feature]==categories[i]]
                y_split[i] = y[X[:,split_feature]==categories[i]]
            column = [split_feature,types[split_feature],categories]
        else:
            median = np.median(X[:,split_feature])
            # print(median)
            X_split = [None]*2
            y_split = [None]*2
            X_split[0] = X[X[:,split_feature]<=median]
            y_split[0] = y[X[:,split_feature]<=median]
            X_split[1] = X[X[:,split_feature]>median]   
            y_split[1] = y[X[:,split_feature]>median]
            column = [split_feature,types[split_feature],median]

        return column,X_split,y_split,np.max(mutual_info) #Return the feature to split on and the splits
    
    def mutual_information_cat(self, x: np.array, y: np.array):
        '''
        Calculates mutual information between x and y
        Args:
            x: numpy array of data [num_samples, 1]
            y: numpy array of classes [num_samples, 1]
        Returns:
            mi: mutual information between x and y
        '''
        H_y = -np.sum([np.sum(y==i)/y.shape[0] * np.log2(np.sum(y==i)/y.shape[0]) for i in np.unique(y)])
        mi  = H_y
        categories = np.unique(x)
        if categories.shape[0] == 1:
            return -1
        for i in categories:
            H_y_x = -np.sum([np.sum(y[x==i]==j)/np.sum(x==i) * np.log2(np.sum(y[x==i]==j)/np.sum(x==i)) for j in np.unique(y[x==i])])
            mi -= np.sum(x==i)/x.shape[0] * H_y_x
        return mi

    def mutual_information_cont(self, x: np.array, y: np.array):
        '''
        Calculates mutual information between x and y
        Args:
            x: numpy array of data [num_samples, 1]
            y: numpy array of classes [num_samples, 1]
        Returns:
            mi: mutual information between x and y
        '''
        H_y = -np.sum([np.sum(y==i)/y.shape[0] * np.log2(np.sum(y==i)/y.shape[0]) for i in np.unique(y)])
        mi  = H_y
        median = np.median(x)
        # if y[x<=median].size == 0 or y[x>median].size == 0:
        #     return -1
        H_y_x_less = -np.sum([np.sum(y[x<=median]==i)/np.sum(x<=median) * np.log2(np.sum(y[x<=median]==i)/np.sum(x<=median)) for i in np.unique(y[x<=median])])
        mi -= np.sum(x<=median)/x.shape[0] * H_y_x_less
        H_y_x_more = -np.sum([np.sum(y[x>median]==i)/np.sum(x>median) * np.log2(np.sum(y[x>median]==i)/np.sum(x>median)) for i in np.unique(y[x>median])])
        mi -= np.sum(x>median)/x.shape[0] * H_y_x_more
        return mi


    def __call__(self, X):
        '''
        Predicted classes for X
        Args:
            X: numpy array of data [num_samples, num_features]
        Returns:
            y: [num_samples, 1] predicted classes
        '''
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y[i] = self.predict(X[i])
        return y.reshape(X.shape[0],1)
    
    def predict(self,x): # Check later
        '''
        Predicted class for x (a single input)
        Args:
            x: numpy array of data [num_features]
        Returns:
            y: [1] predicted class
        '''
        #Note: node.column[0] = feature/column to split on
        #Note: node.column[1] = type of feature/column to split on (cat or cont)
        #Note: node.column[2] = value to split on (for cont) or possible categories (for cat)
        node = self.root
        while not node.is_leaf:
            if node.column is None:
                return node.value
            if node.column[1] == 'cat':
                for i in range(len(node.column[2])):
                    if x[node.column[0]] == node.column[2][i]: 
                        node = node.children[i]
                        break
                else: #Not in any of the categories classified
                    return node.value
            else:
                node = node.children[int(x[node.column[0]]>node.column[2])]
        return node.value
    
    def post_prune(self, X_val, y_val):
        pass

def const_prediction(X_train,y_train,X_test,y_test):
    #Win only prediction    
    #Training set
    y_pred = np.full((y_train.shape[0],1),1)
    accuracy = 100*(np.sum(y_pred==y_train)/y_train.shape[0])
    print(f"Accuracy for in prediction type of only win on training set is {accuracy:.4f}")

    #Test set
    y_pred = np.full((y_test.shape[0],1),1)
    accuracy = 100*(np.sum(y_pred==y_test)/y_test.shape[0])
    print(f"Accuracy for in prediction type of only win on test set is {accuracy:.4f}")

    #Loss only prediction
    #Training set
    y_pred = np.full((y_train.shape[0],1),0)
    accuracy = 100*(np.sum(y_pred==y_train)/y_train.shape[0])
    print(f"Accuracy for in prediction type of only loss on training set is {accuracy:.4f}")

    #Test set
    y_pred = np.full((y_test.shape[0],1),0)
    accuracy = 100*(np.sum(y_pred==y_test)/y_test.shape[0])
    print(f"Accuracy for in prediction type of only loss on test set is {accuracy:.4f}")


def plot_graphs(X_train,y_train,X_test,y_test,types,depths:list[int]):
    train_accuracies = []
    test_accuracies = []
    for depth in depths:
        tree = DTTree()
        tree.fit(X_train,y_train,types,0,depth)
        train_accuracies.append(100*np.sum(tree.__call__(X_train)==y_train)/y_train.shape[0])
        test_accuracies.append(100*np.sum(tree.__call__(X_test)==y_test)/y_test.shape[0])
    for i in range(len(depths)):
        print(f"Accuracy for depth {depths[i]} on training set is {train_accuracies[i]:.4f}")
    for i in range(len(depths)):
        print(f"Accuracy for depth {depths[i]} on test set is {test_accuracies[i]:.4f}")    
    
    #Plot graphs seperately to show the minute differences:
    plt.plot(depths,train_accuracies,label = 'train')
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    plt.plot(depths,test_accuracies,label = 'test')
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


    #Plot graphs together to show overall picture:
    plt.plot(depths,train_accuracies,label = 'train')
    plt.plot(depths,test_accuracies,label = 'test')
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    #change the path if you want
    X_train,y_train = get_np_array(os.path.join(dirname,'../Data/a3_data_starter_code/train.csv'))
    X_test, y_test = get_np_array(os.path.join(dirname,"../Data/a3_data_starter_code/test.csv"))

    # print(X_train,y_train.ravel())
    types = ['cat','cat','cat',"cat","cat","cont","cat","cat","cat" ,"cont","cont" ,"cont" ]

    # max_depth = 45
    # start = time.time()
    # tree = DTTree()
    # tree.fit(X_train,y_train.ravel(),types,0,max_depth = max_depth)
    # y_predicted = tree(X_train)
    # print(f"Prediction accuracy = {np.sum(y_predicted==y_train)/y_train.shape[0]}")
    # print(f"time taken to train + predict: {time.time()-start} seconds")
    
    const_prediction(X_train,y_train,X_test,y_test)
    plot_graphs(X_train,y_train,X_test,y_test,types,[5,10,15,20,25])


'''
Output obtained:

Only Win:
    Accuracy for in prediction type of only win on training set is 50.3380
    Accuracy for in prediction type of only win on test set is 49.6380
Only Lost:
    Accuracy for in prediction type of only loss on training set is 49.6614
    Accuracy for in prediction type of only loss on test set is 50.3619
DT with varying depths on training set:
    Accuracy for depth 5 on training set is 88.5652
    Accuracy for depth 10 on training set is 99.7828
    Accuracy for depth 15 on training set is 99.8466
    Accuracy for depth 20 on training set is 99.8466
    Accuracy for depth 25 on training set is 99.8466
    Accuracy for depth 30 on training set is 99.8466
DT with varying depths on test set:
    Accuracy for depth 5 on test set is 57.6008
    Accuracy for depth 10 on test set is 60.1861
    Accuracy for depth 15 on test set is 60.1861
    Accuracy for depth 20 on test set is 60.1861
    Accuracy for depth 25 on test set is 60.1861
    Accuracy for depth 30 on test set is 60.1861
'''