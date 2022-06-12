import numpy as np
from layers import Layer

class MSELoss(Layer):
    """
    Implementation of Mean Squared Error function
    """
    def __init__(self):
        super(MSELoss, self).__init__()

    def __call__(self, output, target, save_inp=False):
        "Applies Mean Square Loss function on model output and target values"
        if save_inp:
            self.set_inp((output, target))
        return (np.square(output - target)).mean()

    def backward(self):
        "gradient of loss function"
        output, target = self.inp
        grad_out = -2*(target-output)/output.shape[0]
        return grad_out