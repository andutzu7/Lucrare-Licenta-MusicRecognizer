import numpy as np
from .utils import *
import math


"""
2D Convolution is the operation of applying image filters over the initial input, in order to obtain a pixel summed output image.
    
The Conv2D class contains:
    :param n_filters(int): The number of filters that will convolve over the input matrix. The number of channels
    of the output shape.
    :param filter_shape(tuple): A tuple (filter_height, filter_width).
    :param input_shape(tuple): The shape of the expected input of the layer. (batch_size, channels, height, width)
    :param  padding(string): Either 'same' or 'valid'. 'same' results in padding being added so that the output height and width
    matches the input height and width. For 'valid' no padding is added.
    :param stride(int):The stride length of the filters during the convolution over the input.

Sources:    *https://databricks.com/glossary/convolutional-layer
            *https://medium.com/@aakashpydi/implementing-2d-convolution-ebad23f1e43c
            *https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
            *https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
            *https://github.com/eriklindernoren/ML-From-Scratch#machine-learning-from-scratch

"""
class Conv2D:
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

    def forward(self, inputs):
        """
        Performs the forward pass by multipyling the weights with the inputs then adding the biases.
        Args:
            inputs (np.array): given inputs.

        Sources:    *https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
                    *https://github.com/eriklindernoren/ML-From-Scratch#machine-learning-from-scratch
        """
        # Initializing the weights at the first pass (the only time weights can be None)
        if not isinstance(self.weights,np.ndarray):
            # Initializing the input channels size
            self.channel_inputs = inputs.shape[1]
            # Computing the weights value limit
            limit = 1 / math.sqrt(np.prod(self.filter_shape))
            # Computing the weights shape
            weights_size = (self.n_filters,self.channel_inputs, self.filter_shape[0], self.filter_shape[1])
            # Randomly generating the weights matrix
            self.weights = np.random.uniform(-limit, limit, size=weights_size)
        # Asigning the input shape as the input's input shape without the batch size(shape[0])
        self.input_shape = inputs.shape[1:]
        # Extracting the batch size
        batch_size, channels, height, width = inputs.shape
        # Assigning inputs to the layer
        self.layer_input = inputs
        # Turn the input image shape into column shape by stretching the image in regards to the filter shape,stride and padding
        self.X_col = image_to_column(inputs, self.filter_shape, stride=self.stride, output_shape=self.padding)
        # Reshaping the weights to columns
        self.weights_col = self.weights.reshape((self.n_filters, -1))
        # Compute the output
        output = np.dot(self.weights_col,self.X_col) + self.biases
        # Reshape the output to channels_first
        output = output.reshape(self.output_shape() + (batch_size, ))
        # Reshape the output to the initial shape
        output = output.transpose(3,0,1,2)
        # Save the output
        self.output = output

    def backward(self, derivated_values):
        """
        Performs the backward pass using the gradient chaining method.

        Args:
            derivated_values (np.array): Derivated inputs .

        Sources: *https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
                 *https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/layers.py
        """
        # Reshape the inputs to channels first column matrix
        derivated_values = derivated_values.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)

        # Calculated the gradient value
        self.derivated_weights = derivated_values.dot(self.X_col.T).reshape(self.weights.shape)

        self.derivated_biases = np.sum(derivated_values, axis=1, keepdims=True)


        derivated_values = self.weights_col.T.dot(derivated_values)
        
        # Converting the flattened matrix to the (batch_size,channels,height,width) form
        # by performing the inverse operation of img_to_column
        self.derivated_inputs = column_to_image(derivated_values,
                                self.layer_input.shape,
                                self.filter_shape,
                                stride=self.stride,
                                output_shape=self.padding)

    def output_shape(self):
        """
        The output_shape computes the new shape of the input after the forward pass is performed.

        Args:

        Sources: *https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
                 *https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/layers.py
        """
        # Initializing the height and weight
        channels, height, width = self.input_shape
        # Compute the current layer's padding
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        # Compute the new height by adding the padding height to the initial height, substracting the filter height and then dividing 
        # with the stride
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride + 1
        # Performing the same operation for the output's height
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride + 1
        return self.n_filters, int(output_height), int(output_width)

    def get_parameters(self):
        """
        Return the layer's current parameters. 

        Sources: *https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
                 *https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/layers.py
        """
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)