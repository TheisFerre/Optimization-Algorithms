import numpy as np 
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

X_train = X[0: int(len(X) * 0.8)]
X_test = X[int(len(X) * 0.8):]

y_train= y[0: int(len(y) * 0.8)]
y_test = y[int(len(y) * 0.8):]


class NeuralNetwork:

    def __init__(self, input_size, hidden_nodes):

        # Initialize weights normally distributed with mean 0, variance 3
        weights_dict = {}
        for idx, num in enumerate(hidden_nodes):
            if idx == 0:
                weights_dict[f'weights_{idx}'] = np.random.normal(size=(input_size, num)) * 3
                weights_dict[f'bias_{idx}'] = np.random.normal(size=(1, num)) * 3
            else:
                weights_dict[f'weights_{idx}'] = np.random.normal(size=(hidden_nodes[idx-1], num)) * 3
                weights_dict[f'bias_{idx}'] = np.random.normal(size=(1, num)) * 3
        
        self.weights = weights_dict
        self.num_hidden_layers = len(hidden_nodes)
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_deriv(self, x):
        return x * (1-x)
    
    def predict(self, X):

        z = X
        for i in range(self.num_hidden_layers):
            a = np.dot(z, self.weights[f'weights_{i}']) + self.weights[f'bias_{i}']
            z = self.sigmoid(a)
        return z
    
    def loss(self, y_hat, y):
        loss = - (y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
        return loss.mean()
    
    def fit(self, X, y, epochs=50, batch_size=None, learning_rate=0.01):

        if batch_size is None:
            batch_size = len(X)
        
        for _ in range(epochs):

            print(self.loss(self.predict(X), y))

            # Shuffle and divide into batches
            shuffle_idx = np.random.permutation(len(X))
            X_shuffle = X[shuffle_idx]
            y_shuffle = y[shuffle_idx]

            # Compute splits
            num_splits = np.ceil(len(X) / batch_size)
            X_batches = np.array_split(X_shuffle, num_splits)
            y_batches = np.array_split(y_shuffle, num_splits)

            # Tuples of X's and y's
            batches = zip(X_batches, y_batches)

            for batch in batches:
                x_batch, y_batch = batch
                batch_size = len(y_batch)

                # Forward Step
                forward_dict = {f'a_{0}': x_batch, f'z_{0}': x_batch}
                for k in range(self.num_hidden_layers):
                    forward_dict[f'a_{k+1}'] = np.dot(forward_dict[f'z_{k}'], self.weights[f'weights_{k}']) + self.weights[f'bias_{k}']
                    forward_dict[f'z_{k+1}'] = self.sigmoid(forward_dict[f'a_{k+1}'])

                
                # Backprop Step
                deltas = dict()
                derivs = dict()
                for k in range(self.num_hidden_layers-1, -1, -1):

                    if k == self.num_hidden_layers-1:
                        deltas[k] = forward_dict[f'z_{k+1}'] - y_batch.reshape(batch_size, -1)
                    else:
                        deltas[k] = deltas[k+1].dot(self.weights[f'weights_{k+1}'].T) * self.sigmoid_deriv(forward_dict[f'z_{k+1}'])
                    
                    # Compute derivative of error wrt. weights/bias
                    derivs[f'weights_{k}'] = 1/batch_size * forward_dict[f'z_{k}'].T.dot(deltas[k])
                    derivs[f'bias_{k}'] = 1/batch_size * np.sum(deltas[k], keepdims=True)

                    # Update weights
                    self.weights[f'weights_{k}'] = self.weights[f'weights_{k}'] - learning_rate * derivs[f'weights_{k}']
                    self.weights[f'bias_{k}'] = self.weights[f'bias_{k}'] - learning_rate * derivs[f'bias_{k}']





nn = NeuralNetwork(input_size=2, hidden_nodes=[250, 250, 1])

nn.fit(X_train, y_train, batch_size=64, epochs=500, learning_rate=0.001)

# Trivial 0.5 threshold in prediction...
preds_train = np.round(nn.predict(X_train), 0)
preds_test = np.round(nn.predict(X_test), 0)
print(f'Training accuracy:{accuracy_score(y_train, preds_train)}')
print(f'Test accuracy: {accuracy_score(y_test, preds_test)}')



### PLOT DECISION BOUNDARY
xx = np.linspace(min(X[:,0]), max(X[:,0]), num=100)
yy = np.linspace(min(X[:,1]), max(X[:,1]), num=100)

xv, yv = np.meshgrid(xx, yy)

zz = np.array([[nn.predict(np.vstack((xv[j,i], yv[j,i])).T ) for i in range(len(xv))] for j in range(len(yv))])
zz = np.squeeze(zz)

X_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
X_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])

fig, ax = plt.subplots()

cont = ax.contourf(xv, yv, zz, cmap='RdBu_r')
ax.scatter(X_0[:,0], X_0[:,1], c='b')
ax.scatter(X_1[:,0], X_1[:,1], c='r')
fig.colorbar(cont, ax=ax)
plt.savefig('nn_decisionboundary.png')

