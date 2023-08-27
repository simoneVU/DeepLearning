import numpy as np
from datasets.dataset import load_mnist
from models.simple_nn_MNIST import initialize_param, forward, criterion, backward, update_params
import matplotlib.pyplot as plt

(xtrain, ytrain), (xtest, ytest), num_cls = load_mnist(final=True)

costs_train = []
costs_val = []
losses = []
val_losses = []
acc = []
losses_std = []
losses_random = []
losses_random_std = []
epochs = 5
batch_size = 50

#for random_init in  range(5):
params = initialize_param(xtrain.shape[1], 300, num_cls)
  #      losses = []
   #     losses_std = []
print("Start training...")
for epoch in range(epochs):
                costs_train = []
                costs_val = []
                true_predictions = 0
                print(epoch)
                for i in range(0,len(xtrain), batch_size): 
                        dW1_batches, db1_batches, dW2_batches, db2_batches, costs_train, costs_val = [],[],[],[],[],[]
                        for batch in range(batch_size):            
                                Z1,A1,Z2,A2 = forward(np.array([xtrain[i + batch]/255]), params)
                                costs_train.append(criterion(A2, ytrain[i+batch]))
                                grads = backward(np.array([xtrain[i + batch]/255]), params['W2'], A1, A2, ytrain[i + batch])
                                dW1_batches.append(grads['dW1'])
                                db1_batches.append(grads['db1'])
                                dW2_batches.append(grads['dW2'])
                                db2_batches.append(grads['db2'])

                        grads = {"dW1": np.mean(dW1_batches, axis = 0),
                                "db1": np.mean(db1_batches,axis = 0),
                                "dW2": np.mean(dW2_batches, axis = 0),
                                "db2": np.mean(db2_batches,axis = 0)}
                        params = update_params(params, grads, learning_rate = 0.03)

                        #Uncomment to validate
                        #for i in range(0, len(xval), batch_size):
                        #        Z1,A1,Z2,A2 = forward(np.array([xval[i + batch]/255]), params)
                        #        costs_val.append(criterion(A2, yval[i+batch]))

                        #Evaluate
                        for i in range(len(xtest)):
                                Z1,A1,Z2,A2 = forward(np.array([xtest[i]/255]),params)
                                if np.argmax(A2) == ytest[i]:
                                                true_predictions += 1
                        #print(f'Epoch {epoch + 1} : Train -> {np.mean(costs_train)}, Valid -> {np.mean(costs_val)}')

                acc.append(true_predictions/len(ytest))
                #val_losses.append(np.mean(costs_val))
                losses.append(np.mean(costs_train))
                print(f'Epoch {epoch + 1} : Train -> {np.mean(costs_train)} +- {np.std(costs_train)}, Valid -> {acc[epoch]}')
        
        #losses_lr.append(losses)
print("End training...")

plt.plot(losses, label = "train loss")
plt.plot(val_losses, label = "validation loss")
plt.xticks(np.arange(0,5))
plt.ylabel("Loss")
plt.xlabel('Epochs')
plt.title("Averaged Loss over Epochs with MB GD")
plt.legend()  
plt.savefig('images/MNIST/loss_5epochs_MB_GD.png', bbox_inches='tight')