import pandas as pd
from sklearn.datasets import load_boston

from layers import FCLayer
from activations import *
from loss import MSELoss
from model import Model
from utils import standardize, train_test_split, mse


# Training Network
def get_model(inp_dim, lr, activation='tanh'):
    model = Model()
    model.add(FCLayer(inp_dim, 15))
    model.add(ActivationLayer(activation))
    model.add(FCLayer(15, 1))
    
    model.setup(lr=lr, loss_fn=MSELoss())
    return model


if __name__ == '__main__':
    # load data
    housing = load_boston()
    data = pd.DataFrame(housing["data"])
    target = pd.DataFrame(housing["target"])

    # standardize and split data into train and test
    std_data = standardize(data)
    (train_x, train_y), (test_x, test_y) = train_test_split(std_data, target)
    
    # learning
    ACTIVATION = 'tanh'
    N_EPOCHS = 1000
    LR = 0.001
    BS = 64
    model = get_model(train_x.shape[1], LR, ACTIVATION)
    train_error = model.train(
                        train_x.to_numpy(), 
                        train_y.to_numpy(), 
                        batch_size=BS, 
                        n_epochs=N_EPOCHS, 
                        verbose=0
                    )
    print("Model Accuracy on test: {:0.4f}".format(
                                mse(model.predict(test_x.to_numpy()), test_y.to_numpy()))
                            )