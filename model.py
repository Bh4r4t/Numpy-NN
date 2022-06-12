import numpy as np
from tqdm import tqdm
from utils import get_batches

class Model:
    """
    Neural Network Model class
    
    - Define a NN model
    - Train and test NN model on datasets
    
    """
    def __init__(self):
        self.layers = []
        self.lr = None
        self.loss_fn = None

    def setup(self, lr, loss_fn):
        "Setup function for the model"
        self.lr = lr
        self.loss_fn = loss_fn

    def add(self, layer):
        "Utility to add layer to model"
        self.layers.append(layer)

    def forward(self, x, save_inp=False):
        "Forward propagation"
        out = x
        for layer in self.layers:
            out = layer(out, save_inp)
        return out

    def reset_layers(self):
        "Method to clear the gradients and inputs stored in layers"
        for layer in self.layers:
            layer.reset_grads()
            layer.reset_inp()

        self.loss_fn.reset_grads()
        self.loss_fn.reset_inp()

    def backward(self):
        "Backward propagation after loss calculation"
        grad_inp = self.loss_fn.backward()
        # print(grad_inp)
        for layer in reversed(self.layers):
            grad_inp = layer.backward(grad_inp)

    def update(self):
        "Update weights of layers based on gradients computed"
        for layer in self.layers:
            # only update weights for FCLayers
            if layer.grad is not None:
                w_grad, b_grad = layer.grad
                if w_grad is not None:
                    layer.weight -= self.lr*w_grad
                if b_grad is not None:
                    layer.bias -= self.lr*b_grad

    def predict(self, X):
        "Method to run model on test data"
        return self.forward(X)

    def train(self, X, Y, batch_size=None, n_epochs=50, verbose=False, log_freq=10):
        """
        Training function for model
        
        Args:
            X, Y : training data
            batch_size : batch size for batch training
            n_epochs : number of training iterations
            verbose : (0, 1)
            log_freq : loss print frequency
        """
        train_loss = []
        pbar = tqdm(range(int(n_epochs)), total=int(n_epochs), leave=False)
        for epoch in pbar:
            train_batches = get_batches(X, Y, batch_size)
            losses = []
            accuracies = []
            for batch in train_batches:
                self.reset_layers()

                x, y = batch
                # print("batch: ", x.shape, y.shape)
                out = self.forward(x, save_inp=True)
                loss = self.loss_fn(out, y, save_inp=True)
                #print("loss", loss)
                self.backward()
                self.update()
                # print()

                losses.append(loss)
            if verbose and (epoch+1)%log_freq==0:
                print("Epoch: {} | loss: {:0.5f}".format(epoch+1, np.mean(losses)))
            train_loss.append(np.mean(losses))

        return train_loss