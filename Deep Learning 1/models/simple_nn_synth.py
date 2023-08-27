import math 
import random

def forward(X,parameters):  
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = [X[row]*W1[row][col] + X[row+1]*W1[row + 1][col] + b1[col] for row in range(len(W1) - 1) for col in range(len(W1[row]))]
    A1 = [1/(1 + math.exp(-z)) for z in Z1]
    Z2 = [A1[row]*W2[row][col]+ A1[row+1]*W2[row + 1][col] + A1[row+2]*W2[row + 2][col] + b2[col] for row in range(len(W2) - 2) for col in range(len(W2[row]))]
    A2 = [math.exp(z - max(Z2))/sum([math.exp(z - max(Z2)) for z in Z2]) for z in Z2]

    return Z1,A1,Z2,A2

def backward(X, W2, A1, A2, Y):
    if  Y == 0:
        dZ2 = [-1 + A2[0], A2[1]]
    elif Y == 1:
        dZ2 = [A2[0], -1 + A2[1]]
    
    dW2 = [[A1[row]*dZ2[col] for col in range(len(dZ2))] for row in range(len(A1))]
    db2 = dZ2
    dA1 = [a1*(1-a1) for a1 in A1]
    dZ1 = [[sum([dZ2[row]*col[row] for row in range(len(dZ2))]) for col in W2][i]*dA1[i] for i in range(len(dA1))]
    dW1 = [[X[row]*dZ1[col] for col in range(len(dZ1))] for row in range(len(X))]
    db1 = dZ1

        
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

def criterion(A2,Y):
        return -math.log(A2[Y])


def initialize_param(input_shape, hidden_layer_shape, output_layer_shape):
    random.seed(12)
    W1 = [[random.gauss(mu=0.0, sigma=1.0) for i in range(hidden_layer_shape)] for k in range(input_shape)]
    b1 = [0 for i in range(hidden_layer_shape)]
    W2 = [[random.gauss(mu=0.0, sigma=1.0) for i in range(output_layer_shape)] for k in range(hidden_layer_shape)] 
    b2 = [0 for i in range(output_layer_shape)]
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def update_params(parameters, grads, learning_rate):

    W1 = parameters["W1"] 
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = [[w1 - dw1*learning_rate for w1,dw1 in zip(W1[i], dW1[i])] for i in range(len(W1))]
    b1 = [b1 - db1*learning_rate for b1,db1 in zip(b1, db1)] 
    W2 = [[w2 - dw2*learning_rate for w2,dw2 in zip(W2[i], dW2[i])] for i in range(len(W2))]
    b2 = [b2 - db2*learning_rate for b2,db2 in zip(b2, db2)] 
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters