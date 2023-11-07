import numpy as np 
import sys
import os
import math
import time
import pdb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder

def get_data(x_path, y_path):
    '''
    Args:
        x_path: path to x file
        y_path: path to y file
    Returns:
        x: np array of [NUM_OF_SAMPLES x n]
        y: np array of [NUM_OF_SAMPLES]
    '''
    x = np.load(x_path)
    y = np.load(y_path)

    y = y.astype('float')
    x = x.astype('float')

    #normalize x:
    x = 2*(0.5 - x/255)
    # x -= np.mean(x)
    # x /= np.std(x)
    return x, y

def get_metric(y_true, y_pred):
    '''
    Args:
        y_true: np array of [NUM_SAMPLES x r] (one hot) 
                or np array of [NUM_SAMPLES]
        y_pred: np array of [NUM_SAMPLES x r] (one hot) 
                or np array of [NUM_SAMPLES]
                
    '''
    results = classification_report(y_pred, y_true)
    print(results)


class layer:
    def __init__(self,n_units,layer_num=0,up_layer=None,down_layer=None):
        self.n_units = n_units #number of units in layer
        self.layer_num = layer_num
        self.up_layer = up_layer
        self.down_layer = down_layer
        self.theta = None #handling both weight and bias in one matrix -> n_units * (prev_n_units + 1) matrix [+1 for the x0 = 1 term]
        self.input = None #input to layer (would already have the x0 = 1 term) -> (M x (prev_n_units + 1)) matrix -> basically down_layer.output
        self.output = None #output of layer -> (M x n_units) matrix  -> basically activation(input * theta.T)
        self.net = None #net for layer -> (M x n_units) matrix -> basically input * theta.T
        self.delta = None  #delta for layer -> NOT SURE OF THIS --- (M x n_units) matrix -> basically error * activation_derivative(net)
        self.grad_theta = None #gradient of theta for layer -> NOT SURE OF THIS --- (n_units x (prev_n_units + 1)) matrix -> basically delta.T * input
        self.prob_output = None #probability output of  output layer -> (M x n_units) matrix -> basically softmax(input * theta.T)

class neural_network:
    def __init__(self,mini_batch_size,n_features,hidden_layer,target_class,act_func='sigmoid',learning_rate="constant",learning_rate_init=0.01):
        self.start = None #Starting layer - input layer
        self.end = None #Ending layer - output layer
        self.M = mini_batch_size
        self.n = n_features
        self.hidden_layer = hidden_layer
        self.r = target_class
        self.act_func = act_func
        self.learning_rate_type = learning_rate
        self.learning_rate_init = learning_rate_init
        self.layers = []
        self.epochs = 0

    def init_theta(self,layer):
        if layer.layer_num == 0:
            weight = np.random.normal(0,math.sqrt(1/self.n),(layer.n_units,self.n))
            bias = np.zeros((layer.n_units,1))
            layer.theta = np.concatenate((bias,weight),axis=1)
        else:
            weight = np.random.normal(0,math.sqrt(1/layer.up_layer.n_units),(layer.n_units,layer.up_layer.n_units))
            bias = np.zeros((layer.n_units,1))
            layer.theta = np.concatenate((bias,weight),axis=1)
        

    def init_network(self):
        #init input layer
        input_layer = layer(self.hidden_layer[0],layer_num=0,up_layer = None)
        self.init_theta(input_layer)
        self.layers.append(input_layer)
        #init hidden layers
        for i in range(len(self.hidden_layer)-1):
            hidden_layer = layer(self.hidden_layer[i+1],layer_num=i+1,up_layer=self.layers[-1])
            self.init_theta(hidden_layer)
            self.layers.append(hidden_layer)
        #init output layer
        output_layer = layer(self.r,layer_num=len(self.hidden_layer)+1,up_layer=self.layers[-1],down_layer=None)
        self.init_theta(output_layer)
        self.layers.append(output_layer)
        #Set the down_layer for each layer
        for i in range(len(self.layers)-1):
            self.layers[i].down_layer = self.layers[i+1]
        #set start and end layer
        self.start = self.layers[0]
        self.end = self.layers[-1]

    def add_bias(self,x):
        bias = np.ones((x.shape[0],1))
        return np.concatenate((bias,x),axis=1)
    
    def activation(self,x,theta):
        if self.act_func == 'sigmoid':
            net = np.dot(x,theta.T) #size - M x n_units
            output = 1/(1+np.exp(-net)) #size - M x (n_units)
        elif self.act_func == 'relu':
            net = np.dot(x,theta.T) #size - M x n_units
            output = np.maximum(net,0) #size - M x (n_units)
        return output,net 

    def activation_derivative(self,layer):
        if self.act_func == 'sigmoid':
            return layer.output*(1-layer.output) #size - M x n_units
        elif self.act_func == 'relu':
            return np.heaviside(layer.net,0.01) #size - M x n_units

    def softmax(self,x,theta):
        net = np.dot(x,theta.T) #size - M X n_units (specifically in our case Mx5)
        temp = np.exp(net)
        output = temp/temp.sum(axis=1,keepdims = True) #size - M x n_units (specifically Mx5)
        return output,net
    
    def forward_prop(self,x):
        #add bias to input
        x = self.add_bias(x)
        #set input for first layer
        self.start.input = x
        #forward prop
        for layer in self.layers:
            if layer!=self.end: #checking if the layer has a non None down layer (i.e. basically avoiding output layer)
                layer.output,layer.net = self.activation(layer.input,layer.theta) #output 
                layer.down_layer.input = self.add_bias(layer.output) #set input for next layer
            else: #last layer
                layer.prob_output,layer.net = self.softmax(layer.input,layer.theta)
                # layer.output = np.argmax(layer.prob_output,axis=1) + 1

    def backward_prop(self,y): #NOTE - y is one hot encoded
        self.end.delta = self.end.prob_output - y
        #print(self.end.delta) #size - M x r
        self.end.grad_theta = np.dot(self.end.delta.T,self.end.input) #size - r x (prev_n_units+1)
        self.end.grad_theta = self.end.grad_theta/self.M
        for layer in reversed(self.layers[:-1]):
            # layer.delta = np.dot(layer.down_layer.delta,layer.down_layer.theta[:,1:]) * self.activation_derivative(layer) #size - M x n_units
            layer.delta = (layer.down_layer.delta@layer.down_layer.theta[:,1:]) * self.activation_derivative(layer) 
            # print(layer.delta.shape)
            layer.grad_theta = np.dot(layer.delta.T,layer.input) #size - n_units x (prev_n_units+1)
            layer.grad_theta = layer.grad_theta/self.M

    def update_theta(self):
        if self.learning_rate_type == "constant":
            for layer in self.layers:
                layer.theta = layer.theta - self.learning_rate_init*layer.grad_theta
        elif self.learning_rate_type == "adaptive":
            for layer in self.layers:
                layer.theta = layer.theta - self.learning_rate_init*layer.grad_theta/(np.sqrt(self.epochs))

    def SGD_train(self,x,y,y_onehot,max_epochs=100):
        #Perform SGD until convergence or until max_epochs limit
        self.init_network()
        self.epochs = 0
        no_of_batches = 0
        is_converged = False
        while self.epochs < max_epochs:
            #shuffle data
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x_shuffled = x[indices]
            y_oh_shuffled = y_onehot[indices]
            self.epochs += 1
            J_sum  = 0
            J_avg = 0
            for batch in range(y_onehot.shape[0]//self.M):
                x_batch = x_shuffled[self.M*batch:self.M*(batch+1)]
                y_batch = y_oh_shuffled[self.M*batch:self.M*(batch+1)]
                self.forward_prop(x_batch)
                self.backward_prop(y_batch)
                self.update_theta()
                J_sum += -np.sum(y_batch*np.log(self.end.prob_output))/self.M
                no_of_batches += 1
                # if (no_of_batches+1)%100 == 0:
                #     J_prev_avg = J_avg
                #     J_avg = J_sum/100
                #     if abs(J_avg - J_prev_avg) < 1e-3:
                #         print("Moving avg J converged")
                #         is_converged = True
                #         break
                #     J_sum = 0
            if is_converged:
                break
            # print(J_avg)
            print("Epochs: ",self.epochs)
            # print(f"Accuracy rn: {100 * np.sum(self.predict(x,y)==y)/len(y)}%")
        self.epochs = 0

    
    def predict(self,x,y_test=None):
        x = self.add_bias(x)
        next_input = x
        output = None
        for layer in self.layers:
            if layer.down_layer:
                output,net = self.activation(next_input,layer.theta)
                next_input = self.add_bias(output)
            else:
                output,net = self.softmax(next_input,layer.theta)
        prediction = np.argmax(output,axis=1) + 1 #+1 because argmax index starts from 0, but predictions start from 1
        return prediction

#Part a is done as above

def part_b(x_train,y_train,y_train_onehot,x_test,y_test,y_test_onehot):
    M = 32
    n = 1024
    hidden_layers = [[1], [5], [10], [50], [100]]
    train_acc = []
    test_acc = []
    unit_list = []
    r = 5
    for units in hidden_layers:
        print("Number of units: ",units[0])
        network = neural_network(M,n,units,r,'sigmoid','constant',0.01)
        network.SGD_train(x_train,y_train,y_train_onehot,max_epochs=200)
        y_pred = network.predict(x_train,y_train)
        accuracy = np.sum(y_pred==y_train)/y_train.size
        train_acc.append(accuracy)
        print("Training metric:")
        get_metric(y_train,y_pred)
        y_pred = network.predict(x_test,y_test)
        accuracy = np.sum(y_pred==y_test)/y_test.size
        test_acc.append(accuracy)
        unit_list.append(units[0])
        print("Testing metric:")
        get_metric(y_test,y_pred)
    plt.plot(unit_list,train_acc,label="Train")
    plt.plot(unit_list,test_acc,label="Test")
    plt.xlabel("Number of units in hidden layer")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of units in hidden layer")
    plt.legend()
    plt.savefig("./Graphs/part_b.png")
    plt.show()


def part_c(x_train,y_train,y_train_onehot,x_test,y_test,y_test_onehot):
    M = 32
    n = 1024
    hidden_layers = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
    train_acc = []
    test_acc = []
    unit_list = []
    r = 5
    for units in hidden_layers:
        print("Number of hidden layers: ",len(units))
        network = neural_network(M,n,units,r,'sigmoid','constant',0.01)
        network.SGD_train(x_train,y_train,y_train_onehot,max_epochs=200)
        y_pred = network.predict(x_train,y_train)
        accuracy = np.sum(y_pred==y_train)/y_train.size
        train_acc.append(accuracy)
        print("Training metric:")
        get_metric(y_train,y_pred)
        y_pred = network.predict(x_test,y_test)
        accuracy = np.sum(y_pred==y_test)/y_test.size
        test_acc.append(accuracy)
        unit_list.append(len(units))
        print("Testing metric:")
        get_metric(y_test,y_pred)
    plt.plot(unit_list,train_acc,label="Train")
    plt.plot(unit_list,test_acc,label="Test")
    plt.xlabel("Number of hidden layers")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of hidden layers")
    plt.legend()
    plt.savefig("./Graphs/part_c.png")
    plt.show()

def part_d(x_train,y_train,y_train_onehot,x_test,y_test,y_test_onehot):
    M = 32
    n = 1024
    hidden_layers = [[512],[512,256],[512, 256, 128], [512, 256, 128, 64]]
    train_acc = []
    test_acc = []
    unit_list = []
    r = 5
    for units in hidden_layers:
        print("Number of hidden layers: ",len(units))
        network = neural_network(M,n,units,r,'sigmoid','adaptive',0.01)
        network.SGD_train(x_train,y_train,y_train_onehot,max_epochs=500)
        y_pred = network.predict(x_train,y_train)
        accuracy = np.sum(y_pred==y_train)/y_train.size
        train_acc.append(accuracy)
        print("Training metric:")
        get_metric(y_train,y_pred)
        y_pred = network.predict(x_test,y_test)
        accuracy = np.sum(y_pred==y_test)/y_test.size
        test_acc.append(accuracy)
        unit_list.append(len(units))
        print("Testing metric:")
        get_metric(y_test,y_pred)
    plt.plot(unit_list,train_acc,label="Train")
    plt.plot(unit_list,test_acc,label="Test")
    plt.xlabel("Number of hidden layers")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of hidden layers")
    plt.legend()
    plt.savefig("./Graphs/part_d.png")
    plt.show()

def part_e(x_train,y_train,y_train_onehot,x_test,y_test,y_test_onehot):
    M = 32
    n = 1024
    hidden_layers = [[512], [512, 256], [512, 256, 128], [512, 256, 128, 64]]
    train_acc = []
    test_acc = []
    unit_list = []
    r = 5
    for units in hidden_layers:
        print("Number of hidden layers: ",len(units))
        network = neural_network(M,n,units,r,'relu','adaptive',0.01)
        network.SGD_train(x_train,y_train,y_train_onehot,max_epochs=200)
        y_pred = network.predict(x_train,y_train)
        accuracy = np.sum(y_pred==y_train)/y_train.size
        train_acc.append(accuracy)
        print("Training metric:")
        get_metric(y_train,y_pred)
        y_pred = network.predict(x_test,y_test)
        accuracy = np.sum(y_pred==y_test)/y_test.size
        test_acc.append(accuracy)
        unit_list.append(len(units))
        print("Testing metric:")
        get_metric(y_test,y_pred)
    plt.plot(unit_list,train_acc,label="Train")
    plt.plot(unit_list,test_acc,label="Test")
    plt.xlabel("Number of hidden layers")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of hidden layers")
    plt.legend()
    plt.savefig("./Graphs/part_e.png")
    plt.show()
    
def part_f(x_train,y_train,y_train_onehot,x_test,y_test,y_test_onehot):
    hidden_layers = [(512), (512, 256), (512, 256, 128), (512, 256, 128, 64)]
    train_acc = []
    test_acc = []
    unit_list = []
    for i,layers in enumerate(hidden_layers):
        print("Number of hidden layers: ",i+1)
        clf = MLPClassifier(hidden_layer_sizes=layers,activation='relu',solver='sgd',batch_size=32,learning_rate='invscaling',max_iter=200)
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_train)
        accuracy = np.sum(y_pred==y_train)/y_train.size
        train_acc.append(accuracy)
        print("Training metric:")
        get_metric(y_train,y_pred)
        y_pred = clf.predict(x_test)
        accuracy = np.sum(y_pred==y_test)/y_test.size
        test_acc.append(accuracy)
        unit_list.append(len(layers))
        print("Testing metric:")
        get_metric(y_test,y_pred)
    plt.plot(unit_list,train_acc,label="Train")
    plt.plot(unit_list,test_acc,label="Test")
    plt.xlabel("Number of hidden layers")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of hidden layers")
    plt.legend()
    plt.savefig("./Graphs/part_f.png")
    plt.show()


if __name__ == '__main__':

    dirname = os.path.dirname(__file__)

    x_train_path = os.path.join(dirname,"../Data/part b/x_train.npy")
    y_train_path = os.path.join(dirname,"../Data/part b/y_train.npy")

    # x_train_path = sys.argv[1]
    # y_train_path = sys.argv[2]

    X_train, y_train = get_data(x_train_path, y_train_path)
    # print(X_train.shape,y_train.shape)

    x_test_path = os.path.join(dirname,"../Data/part b/x_test.npy")
    y_test_path = os.path.join(dirname,"../Data/part b/y_test.npy")

    # x_test_path = sys.argv[3]
    # y_test_path = sys.argv[4]

    X_test, y_test = get_data(x_test_path, y_test_path)
    #you might need one hot encoded y in part a,b,c,d,e
    label_encoder = OneHotEncoder(sparse_output = False)
    label_encoder.fit(np.expand_dims(y_train, axis = -1))

    y_train_onehot = label_encoder.transform(np.expand_dims(y_train, axis = -1))
    y_test_onehot = label_encoder.transform(np.expand_dims(y_test, axis = -1))

    # network = neural_network(32,1024,[512,256,128,64],5,'sigmoid','constant',0.01)
    # network.SGD_train(X_train,y_train,y_train_onehot,max_epochs=200)
    # y_pred = network.predict(X_test,y_test)
    # get_metric(y_test,y_pred)

    # print("Starting part b")
    # part_b(X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot)
    # time.sleep(10)
    # print("Starting part c")
    # part_c(X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot)
    # time.sleep(10)
    # print("Starting part d")
    # part_d(X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot)
    # time.sleep(10)
    # print("Starting part e")
    # part_e(X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot)
    # time.sleep(10)
    # print("Starting part f")
    # part_f(X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot)