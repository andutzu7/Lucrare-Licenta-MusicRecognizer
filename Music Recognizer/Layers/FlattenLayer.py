import numpy as np

class Flatten:
    """ Turns a multidimensional matrix into two-dimensional """
    def __init__(self):
        self.prev_shape = None
        self.trainable = True
        self.derivated_inputs = None
        self.output = None

    def forward(self, inputs,training=True):
        self.input_shape = inputs.shape[1:]
        self.prev_shape = inputs.shape
        self.output = inputs.reshape((inputs.shape[0], -1))

    def backward(self, accum_grad):
        self.derivated_inputs = accum_grad.reshape(self.prev_shape)

    def output_shape(self):
        return (np.prod(self.input_shape),)