import numpy as np
from .utils import *


class MaxPooling2D:
    """
    MaxPooling2D is the operation of extracting the most important features over a given image. The operating way of the MaxPooling
    layer is similar to the Conv2D, applying a pool_filter from stride to stride in order to extract the maximal numerical values.
        
    The MaxPooling2D class contains:
        :param pool_shape(tuple): A tuple containing the size (width height) of the pooling filter
        :param stride(int):The stride length of the filters during the convolution over the input.
        :param  padding(int): The amount of padding to be added at the edges.

    Sources:    *https://deeplizard.com/learn/video/ZjM_XQa5s6s
                *https://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/
                *https://github.com/eriklindernoren/ML-From-Scratch#machine-learning-from-scratch

    """
    def __init__(self, pool_shape=(2, 2), stride=1, padding=0):
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding
        self.trainable = True
        self.derivated_inputs = None
        self.output = None

    def forward(self, inputs):
        """
        Performs the forward pass by performing an argmax calculation over each chunk of the flattened image.

        Args:
            inputs (np.array): given inputs.

        Sources:    *https://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/
                    *https://github.com/eriklindernoren/ML-From-Scratch#machine-learning-from-scratch
        """
        # Saving the inputs
        self.layer_input = inputs
        # Cutting the batch size from the inputs
        self.input_shape = inputs.shape[1:]
        # Retrieving the original shape parameters
        batch_size, channels, height, width = inputs.shape
        # Computing the desired output shape
        _, out_height, out_width = self.output_shape()
        # Reshape the inputs accordingly
        inputs = inputs.reshape(batch_size*channels, 1, height, width)
        # Flattening the image to the column configuration
        X_col = image_to_column(
            inputs, self.pool_shape, self.stride, output_shape="same")
        # Performing the Pooling by extracting the idexes of the values with the max values
        arg_max = np.argmax(X_col, axis=0).flatten()
        # Computing the output by extracting the values at the given idex
        output = X_col[arg_max, range(arg_max.size)]
        # Saving the argmax for the backpropagation
        self.cache = arg_max
        # Reshaping the output and transposing it (N,C,H,W)
        output = output.reshape(out_height, out_width, batch_size, channels)
        output = output.transpose(2, 3, 0, 1)

        self.output = output

    def backward(self, derivated_values):
        """
        Performs the backward pass using the gradient chaining method.

        Args:
            derivated_values (np.array): Derivated inputs .

        Sources:    *https://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/
                    *https://github.com/eriklindernoren/ML-From-Scratch#machine-learning-from-scratch
        """
        # Exstracting the batch size.
        batch_size, _, _, _ = derivated_values.shape
        # Extracting the dimensions of the input
        channels, height, width = self.input_shape
        # Transposing and flattening the values
        derivated_values = derivated_values.transpose(2, 3, 0, 1).ravel()
        # Initializing the flattened output array
        derivated_values_flattened = np.zeros((np.prod(self.pool_shape), derivated_values.size))
        # Retrieving the cache from the forward pass
        arg_max = self.cache
        # Retrieving the max values
        derivated_values_flattened[arg_max, range(derivated_values.size)] = derivated_values
        # Transforming the values back to a 4d array in regards to the flattened result
        derivated_values = column_to_image(
            derivated_values_flattened, (batch_size * channels, 1, height, width), self.pool_shape, self.stride, 'same')
        # Reshaping the values to (N,C,H,W)
        derivated_values = derivated_values.reshape((batch_size,) + self.input_shape)
        self.derivated_inputs = derivated_values

    def output_shape(self):
        """
        The output_shape computes the new shape of the input after the forward pass is performed.

        Args:

        Sources:    *https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/layers.py
        """
        # Exstract the input shape dimensions
        channels, height, width = self.input_shape
        # Computing the current width and height
        out_width = (width + self.stride - 1)//self.stride
        out_height = (height + self.stride - 1)//self.stride
        # Return the output dimensions
        return channels, int(out_height), int(out_width)
        
