import numpy as np
from datasets.dataset import load_mnist
from models.simple_nn_MNIST import initialize_param, forward, criterion, backward, update_params
import matplotlib.pyplot as plt
import random

(xtrain, ytrain), (xtest, ytest), num_cls = load_mnist(final=True)

costs_train = []
losses = []
accuracies = []
acc = []
val = []
loss = []
costs_val = []
losses_lr = []
losses_val = []
epochs = 5
batch_size = 150 


params = initialize_param(xtrain.shape[1], 300, num_cls)
for epoch in range(epochs):
                #Shuffling the lists for SGD
                c, d = list(zip(xtrain, ytrain)),list(zip(xval, yval))
                random.shuffle(c)
                random.shuffle(d)
                xtrain, ytrain = zip(*c)
                xval, yval = zip(*d)
                true_predictions = 0
                for i in range(len(xtrain)):   
                        Z1,A1,Z2,A2 = forward(np.array([xtrain[i]/255]),params)
                        cost = criterion(A2, ytrain[i])
                        grads = backward(np.array([xtrain[i]/255]), params['W2'], A1, A2, ytrain[i])
                        params = update_params(params, grads, learning_rate=0.03)
                        loss.append(cost)
                        if i % batch_size == 0:
                                avg_cost = (sum(loss)/len(loss))
                                loss = []
                                losses.append(avg_cost)
                costs_train.append(np.mean(losses))

                #Validate
                #for i in range(len(xval)):   
                #        Z1,A1,Z2,A2 = forward(np.array([xval[i]/255]),params)
                #        cost = criterion(A2, yval[i])
                #        val.append(cost)
                #        if i % batch_size == 0:
                #                avg_val = (sum(val))/len(val)
                #                val = []
                #                losses_val.append(avg_val)
                #costs_val.append(np.mean(losses_val))

                #Evaluate
                for i in range(len(xtest)):
                       Z1,A1,Z2,A2 = forward(np.array([xtest[i]/255]),params)
                       if np.argmax(A2) == ytest[i]:
                               true_predictions += 1
                
                acc.append(true_predictions/len(ytest))
                  
print("End training...")

plt.plot(acc, label = "train loss 0.03")
plt.ylabel("Accuracy")
plt.xticks([0,1,2,3,4])
plt.xlabel('Epochs')
plt.title("Accuracy over Epochs with SGD")
plt.legend()  
plt.savefig('images/MNIST/loss_5epochs_lr_003.png', bbox_inches='tight')