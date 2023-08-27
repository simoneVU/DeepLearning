from datasets.dataset import load_synth
from models.simple_nn_synth import initialize_param, forward, backward, criterion, update_params
from utils import normalize
import matplotlib.pyplot as plt
    
#Loading data
(xtrain, ytrain), (xval, yval), num_cls = load_synth()

params = initialize_param(xtrain.shape[1], 3, num_cls)
xtrain = normalize(xtrain)
#Shuffling the train data for SGD

loss = []
losses = []
epochs = 5
batch_size = 150

print("Start training...")
#for epoch in range(epochs):#comment this line one for avg. loss over batches
#c = list(zip(xtrain, ytrain))
#random.shuffle(c)
#xtrain, ytrain = zip(*c)

for i in range(len(xtrain)):
                Z1,A1,Z2,A2 = forward(xtrain[i],params)
                cost = criterion(A2, ytrain[i])
                grads = backward(xtrain[i], params['W2'], A1, A2, ytrain[i])
                params = update_params(params, grads, learning_rate=0.01)
                loss.append(cost)
                if i % batch_size == 0:
                    avg_cost = (sum(loss)/len(loss))
                    loss = []
                    losses.append(avg_cost)
                    print(f'Loss epoch {i} {losses[int(i/batch_size)]}')
print("End training...")

plt.plot(losses)
plt.ylabel("Averaged Loss over 150 iterations")
plt.xlabel('150 x iter')
plt.title("Averaged Loss over SGD iter")
plt.savefig('images/loss_batches.png', bbox_inches='tight')