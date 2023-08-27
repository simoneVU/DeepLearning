import numpy as np
from datasets.dataset import load_mnist
from models.simple_nn_MNIST_batched import initialize_param, forward, criterion, backward, update_params
import matplotlib.pyplot as plt

(xtrain, ytrain), (xval, yval), num_cls = load_mnist(final=True)
epochs = 5
random_losses = []
print("Start training...")

#for i in range(3):
params = initialize_param(xtrain.shape[1], 300, num_cls)
loss_train, acc, val = [], [],[]
batch_size = 10
for epoch in range(epochs):
    for batch in range(0,len(xtrain), batch_size):
        grads = {"dW1": 0,"db1":  0,"dW2":  0,"db2":  0}    
        Z1,A1,Z2,A2 = forward(xtrain[batch:batch+batch_size]/255,params)
        cost = criterion(A2, ytrain[batch:batch+batch_size])
        grads = backward(xtrain[batch:batch+batch_size]/255, params['W2'], A1, A2,  ytrain[batch:batch+batch_size])
        params = update_params(params, grads, learning_rate=0.03)
        loss_train.append(cost)
        
        #Validate
        for batch in range(0,len(xval), batch_size):
            Z1,A1,Z2,A2 = forward(xval[batch:batch+batch_size]/255, params) 
            cost = criterion(A2, yval[batch:batch+batch_size])
            val.append(cost)

        #Evaluate 
        #Z1,A1,Z2,A2 = forward(xtest/255, params)          
        #predictions = [np.argmax(a2) for a2 in A2]   
        #test_acc = np.sum([np.argmax(a2) for a2 in A2] == ytest) / ytest.size

        #loss_train.append(cost)
        #loss_val.append(val_loss)
        #acc.append(test_acc)
    print(f'Epoch {epoch + 1} : Train -> {np.mean(loss_train)},  Val-> {np.mean(val)}')
    #random_losses.append(loss_train), Valid-> {val_loss}

print("End training...")

#fig, ax = plt.subplots(1)
#ax.plot(list(range(epochs)), np.mean(random_losses,axis = 0), lw=2, label='avg. loss', color='blue')
#ax.fill_between(list(range(epochs)), np.mean(random_losses,axis = 0)+np.std(random_losses,axis = 0), np.mean(random_losses,axis = 0)-np.std(random_losses,axis = 0), facecolor='blue', alpha=0.5)
#ax.set_title(r'Train loss with random initialization with $\mu$ and $\pm \sigma$ interval and GD')
#ax.legend(loc='upper left')
#ax.set_xlabel('Epoch')
#ax.set_ylabel('Train Loss')
#ax.set_xticks([0,1,2,3,4])
#plt.savefig('images/MNIST/loss_5epochs_random_init_GD.png', bbox_inches='tight')

plt.plot(loss_train, label = "train loss")
plt.plot(acc, label = "test accuracy")
plt.xticks(np.arange(0,5))
plt.ylabel("Train loss and Test Accuracy with GD")
plt.xlabel('Epoch')
plt.title("Loss/Acc ")
plt.legend()  
plt.savefig('images/MNIST/loss_5epochs_GD_test_0.03.png', bbox_inches='tight')