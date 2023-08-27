import numpy as np

def softmax(Z):
    x_max = np.amax(Z, axis=1, keepdims=True)
    exp_x_shifted = np.exp(Z - np.max(Z))
    return exp_x_shifted / np.sum(exp_x_shifted, axis=1, keepdims=True)

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def forward(X,parameters):  
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = X@W1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1@W2 + b2
    A2 = softmax(np.array(Z2,dtype=np.float128)) 
    return Z1,A1,Z2,A2

def backward(X, W2, A1, A2, Y):
    dZ2 = np.array([[a2[i] if i != y else -1 + a2[i] for i in range(0,10)] for a2,y in zip(A2,Y)])
    dW2 = A1.T@dZ2
    db2 = np.sum(dZ2, keepdims = True) 
    dA1 = A1*(1-A1)
    dZ1 = dZ2@W2.T*dA1
    dW1 = X.T@dZ1
    db1 = np.sum(dZ1, keepdims = True) 

    dW2 = np.clip(dW2, -5, 5)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

def criterion(A2,Y):
    cost = np.mean(np.array([[-np.log(a2[y]) for a2,y in zip(A2,Y)]]))
    return cost


def initialize_param(input_shape, hidden_layer_shape, output_layer_shape):
    np.random.seed(12)
    W1 = np.random.randn(input_shape, hidden_layer_shape)*np.sqrt(1/(input_shape+hidden_layer_shape))
    b1 = np.zeros((1, hidden_layer_shape))
    W2 = np.random.randn(hidden_layer_shape, output_layer_shape)*np.sqrt(1/(hidden_layer_shape+output_layer_shape))
    b2 = np.zeros((1, output_layer_shape))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def update_params(parameters, grads, learning_rate):

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    W1 = np.subtract(W1,dW1*learning_rate)
    b1 = np.subtract(b1,db1*learning_rate)
    W2 = np.subtract(W2,dW2*learning_rate)
    b2 = np.subtract(b2,db2*learning_rate)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters