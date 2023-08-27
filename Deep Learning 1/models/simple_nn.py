import math 

def forward(X,W1,b1,W2,b2):  
    Z1 = [X[row]*W1[row][col] + X[row+1]*W1[row + 1][col] + b1[col] for row in range(len(W1) - 1) for col in range(len(W1[row]))]
    A1 =[1/(1 + math.exp(-z)) for z in Z1]
    Z2 = [A1[row]*W2[row][col]+ A1[row+1]*W2[row + 1][col] + A1[row+2]*W2[row + 2][col] + b2[col] for row in range(len(W2) - 2) for col in range(len(W2[row]))]
    A2 = [math.exp(z)/sum([math.exp(z) for z in Z2]) for z in Z2]

    return Z1,A1,Z2,A2

def backward(X, W2, A1, A2, Y):
    dZ2 = [-1 + A2[c]  if c == 1 else A2[c] for c in Y]
    dW2 = [[A1[row]*dZ2[col] for col in range(len(dZ2))] for row in range(len(A1))] 
    db2 = dZ2
    dA1 = [a1*(1-a1) for a1 in A1]
    dZ1 = [[sum([dZ2[row]*col[row] for row in range(len(dZ2))]) for col in W2][i]*dA1[i] for i in range(len(dA1))]
    dW1 = [[X[row]*dZ1[col] for col in range(len(dZ1))] for row in range(len(X))]
    db1 = dZ1

    return dZ1, dW1, db1, dZ2, dW2, db2

def criterion(A2):
    loss = sum([-math.log(a) for a in A2])
    return loss

