import numpy as np

class Layer:
    # Abstract class
    def __init__(self):
        self.reset_grads()
        self.reset_inp()

    def reset_grads(self):
        """
        Stores the gradients of current layer
         - for linear layers grad contains a tuple of (grad_weights, grad_bias)
         - for activation layers grad is None
        """
        self.grad = None

    def reset_inp(self):
        self.inp = None

    def set_inp(self, inp):
        # if self.inp is None:
        #     warnings.warn("Found previous state values. First clear the state using `reset_inp_out()` ")
        assert self.inp is None, "Found previous state values. First clear the state using `reset_inp_out()` "
        self.inp = inp

    def backward(self, grad_inp):
        pass


class FCLayer(Layer):
    """
    MLP Layer.
    Performs : Y = x.w^T + b
    """
    def __init__(self, n_inp, n_out, bias=True):
        super(FCLayer, self).__init__()
        self.weight = np.ones((n_inp, n_out))
        if bias:
            self.bias = np.zeros((1, n_out))
        else:
            self.bias = None
        self.init_parameters()

    def init_parameters(self):
        "Method to initialize weight and bias of current layer"
        n_inp, n_out = self.weight.shape
        self.weight = np.random.uniform(-0.01, 0.01, (n_inp, n_out))

    # Forward Propagation
    def __call__(self, x, save_inp=False):
        out = x @ self.weight
        if isinstance(self.bias, np.ndarray):
            out += self.bias
        if save_inp:
            self.set_inp(x)
        return out

    # Backward Propagation
    def backward(self, grad_inp):
        # gradients for weights
        weight_grad = self.inp.T @ grad_inp
        # gradients for bias
        if isinstance(self.bias, np.ndarray):
            bias_grad = np.sum(grad_inp, 0)
        else:
            bias_grad = None

        self.grad = (weight_grad, bias_grad)

        # gradient flow for previous layers
        grad_out = grad_inp @ self.weight.T
        return grad_out