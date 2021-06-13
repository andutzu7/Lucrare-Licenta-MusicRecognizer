import numpy as np
from .utils import *


class MaxPooling2D:
    """A parent class of MaxPooling2D and AveragePooling2D
    """

    def __init__(self, pool_shape=(2, 2), stride=1, padding=0):
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding
        self.trainable = True
        self.derivated_inputs = None
        self.output = None

    def forward(self, inputs,training=True):
        self.layer_input = inputs

        self.input_shape = inputs.shape[1:]
        batch_size, channels, height, width = inputs.shape
        _, out_height, out_width = self.output_shape()
        inputs = inputs.reshape(batch_size*channels, 1, height, width)
        X_col = image_to_column(
            inputs, self.pool_shape, self.stride, output_shape="same")
        arg_max = np.argmax(X_col, axis=0).flatten()
        output = X_col[arg_max, range(arg_max.size)]
        self.cache = arg_max

        output = output.reshape(out_height, out_width, batch_size, channels)
        output = output.transpose(2, 3, 0, 1)

        self.output = output

    def backward(self, accum_grad):
        batch_size, _, _, _ = accum_grad.shape
        channels, height, width = self.input_shape
        accum_grad = accum_grad.transpose(2, 3, 0, 1).ravel()

        accum_grad_col = np.zeros((np.prod(self.pool_shape), accum_grad.size))
        arg_max = self.cache
        accum_grad_col[arg_max, range(accum_grad.size)] = accum_grad

        accum_grad = column_to_image(
            accum_grad_col, (batch_size * channels, 1, height, width), self.pool_shape, self.stride, 'same')
        accum_grad = accum_grad.reshape((batch_size,) + self.input_shape)

        self.derivated_inputs = accum_grad

    def output_shape(self):
        channels, height, width = self.input_shape
        out_width = (width + self.stride - 1)//self.stride
        out_height = (height + self.stride - 1)//self.stride
        return channels, int(out_height), int(out_width)
