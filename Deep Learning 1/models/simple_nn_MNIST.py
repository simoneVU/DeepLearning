import numpy as np

def softmax(Z):
    return np.exp(Z)/np.sum(np.exp(Z))

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def forward(X,parameters):  
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = ((W1@X.T) +b1)
    A1 = sigmoid(Z1)
    Z2 = W2@A1 + b2
    A2 = softmax(np.array(Z2,dtype=np.float128)) 
    return Z1,A1,Z2,A2

def backward(X, W2, A1, A2, Y):
    dZ2 = np.array([A2[i] if i != Y else -1 + A2[i] for i in range(0,10)])
    dW2 = dZ2@A1.T
    db2 = dZ2
    dA1 = A1*(1-A1)
    dZ1 = W2.T@dZ2*dA1
    dW1 = dZ1@X
    db1 = dZ1

        
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

def criterion(A2,Y):
    return -np.log(A2[Y][0])


def initialize_param(input_shape, hidden_layer_shape, output_layer_shape):
    np.random.seed(12)
    W1 = np.random.rand(hidden_layer_shape,input_shape)*np.sqrt(1/(input_shape+hidden_layer_shape))
    b1 = np.zeros((hidden_layer_shape, 1))
    W2 = np.random.rand(output_layer_shape,hidden_layer_shape)*np.sqrt(1/(input_shape+hidden_layer_shape))
    b2 = np.zeros((output_layer_shape, 1))
    
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

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    W1 = np.subtract(W1,dW1*learning_rate)
    b1 = np.subtract(b1,db1*learning_rate)
    W2 = np.subtract(W2,dW2*learning_rate)
    b2 = np.subtract(b2,db2*learning_rate)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters