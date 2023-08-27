from models.simple_nn import forward, backward, criterion

W1 = [[1., 1., 1.], [-1., -1., -1.]]
b1 = [0., 0., 0.]

W2 = [[1., 1.], [-1., -1.], [-1.,-1.]]
b2 = [0., 0.]

X = [1., -1.]
Y = [1, 0]

Z1,A1,Z2,A2 = forward(X,W1,b1,W2,b2)
cost = criterion(A2)
dZ1, dW1, db1, dZ2, dW2, db2 = backward(X, W2, A1, A2, Y)

print(f'Z1 is {Z1}')
print(f'A1 is {A1}')
print(f'Z2 is {Z2}')
print(f'A2 is {A2}')
print(f'dZ1 is {dZ1}')
print(f'dW1 is {dW1}')
print(f'db1 is {db1}')
print(f'dZ2 is {dZ2}')
print(f'dW2 is {dW2}')
print(f'db2 is {db2}')
print(f'Loss is {cost}')