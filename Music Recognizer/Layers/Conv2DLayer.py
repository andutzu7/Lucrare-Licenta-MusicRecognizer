import numpy as np
from .utils import *
import math


class Conv2D():
    """A 2D Convolution Layer.
    Parameters:
    -----------
    n_filters: int
        The number of filters that will convolve over the input matrix. The number of channels
        of the output shape.
    filter_shape: tuple
        A tuple (filter_height, filter_width).
    input_shape: tuple
        The shape of the expected input of the layer. (batch_size, channels, height, width)
        Only needs to be specified for first layer in the network.
    padding: string
        Either 'same' or 'valid'. 'same' results in padding being added so that the output height and width
        matches the input height and width. For 'valid' no padding is added.
    stride: int
        The stride length of the filters during the convolution over the input.
    """
    def __init__(self, n_filters, filter_shape,channel_inputs=1, padding='same', stride=1):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.channel_inputs = channel_inputs
        self.padding = padding
        self.stride = stride
        self.trainable = True
        self.output = None
        self.derivated_inputs = None
        self.derivated_weights = None
        self.biases = np.zeros((self.n_filters, 1))
        self.weights = None

    def get_parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward(self, inputs):
        '''
        Initializing the weights if its the first pass
        '''
        if not isinstance(self.weights,np.ndarray):
            self.channel_inputs = inputs.shape[1]
            limit = 1 / math.sqrt(np.prod(self.filter_shape))
            w_size = (self.n_filters,self.channel_inputs, self.filter_shape[0], self.filter_shape[1])
            self.weights = np.random.uniform(-limit, limit, size=w_size)
        self.input_shape = inputs.shape[1:]
        batch_size, channels, height, width = inputs.shape
        self.layer_input = inputs
        # Turn image shape into column shape
        # (enables dot product between input and weights)
        self.X_col = image_to_column(inputs, self.filter_shape, stride=self.stride, output_shape=self.padding)
        # Turn weights into column shape
        self.weights_col = self.weights.reshape((self.n_filters, -1))
        # Calculate output
        output = np.dot(self.weights_col,self.X_col) + self.biases
        # Reshape into (n_filters, out_height, out_width, batch_size)
        output = output.reshape(self.output_shape() + (batch_size, ))
        # Redistribute axises so that batch size comes first
        output = output.transpose(3,0,1,2)
        self.output = output
#de gandit asta pt flowul meu
    def backward(self, accum_grad):
        # Reshape accumulated gradient into column shape
        accum_grad = accum_grad.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)

        # Recalculate the gradient which will be propogated back to prev. layer
        self.derivated_weights = accum_grad.dot(self.X_col.T).reshape(self.weights.shape)

        self.derivated_biases = np.sum(accum_grad, axis=1, keepdims=True)


        accum_grad = self.weights_col.T.dot(accum_grad)
        
        self.derivated_inputs = column_to_image(accum_grad,
                                self.layer_input.shape,
                                self.filter_shape,
                                stride=self.stride,
                                output_shape=self.padding)


    def output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride + 1
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride + 1
        return self.n_filters, int(output_height), int(output_width)

