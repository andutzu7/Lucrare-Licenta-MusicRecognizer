import numpy as np

class Flatten():
    """ Turns a multidimensional matrix into two-dimensional """
    def __init__(self, input_shape=None):
        self.prev_shape = None
        self.trainable = True
        self.derivated_inputs = None
        self.output = None

    def forward(self, X):
        self.input_shape = X.shape[1:]
        self.prev_shape = X.shape
        self.output = X.reshape((X.shape[0], -1))

    def backward(self, accum_grad):
        self.derivated_inputs = accum_grad.reshape(self.prev_shape)

    def output_shape(self):
        return (np.prod(self.input_shape),)