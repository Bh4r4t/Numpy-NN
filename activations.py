import numpy as np
from scipy.special import expit
from layers import Layer

class ActivationLayer(Layer):
    """
    Activations function layers.

    Allow usage of `tanh`, `sigmoid` and `relu` activations.
    """
    _activations = ['tanh', 'sigmoid', 'relu']
    def __init__(self, activation):
        super(ActivationLayer, self).__init__()
        self.act_func = None            # activation function
        self.der_act_func = None        # derivative of activation function
        self.set_functions(activation)
    
    def set_functions(self, activation):
        "Sets the activation function and its derivative function"
        assert activation in self._activations, "Invalid Activation function name!"

        if activation == 'tanh':
            self.act_func = lambda x: np.tanh(x)
            self.der_act_func = lambda x: 1.0 - np.square(np.tanh(x))
        elif activation == 'sigmoid':
            self.act_func = lambda x: expit(x)
            self.der_act_func = lambda x: expit(x) * (1.0 - expit(x))
        elif activation == 'relu':
            self.act_func = lambda x: max(0, x)
            self.der_act_func = lambda x: x * (x > 0)

    # Forward Propagation
    def __call__(self, x, save_inp=False):
        out = self.act_func(x)
        if save_inp:
            self.set_inp(x)
        return out

    # Backward Propagation
    def backward(self, grad_inp):
        grad_out = grad_inp*self.der_act_func(self.inp)
        return grad_out