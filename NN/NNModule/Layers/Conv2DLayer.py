import numpy as np
import copy
import math

def column_to_image(cols, images_shape, filter_shape, stride, output_shape='same'):
    batch_size, channels, height, width = images_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    height_padded = height + np.sum(pad_h)
    width_padded = width + np.sum(pad_w)
    images_padded = np.zeros(
        (batch_size, channels, height_padded, width_padded))

    # Calculate the indices where the dot products are applied between weights
    # and the image
    k, i, j = get_im2col_indices(
        images_shape, filter_shape, (pad_h, pad_w), stride)

    cols = cols.reshape(channels * np.prod(filter_shape), -1, batch_size)
    cols = cols.transpose(2, 0, 1)
    # Add column content to the images at the indices
    np.add.at(images_padded, (slice(None), k, i, j), cols)

    # Return image without padding
    return images_padded[:, :, pad_h[0]:height+pad_h[0], pad_w[0]:width+pad_w[0]]


def get_im2col_indices(images_shape, filter_shape, padding, stride=1):

    # First figure out what the size of the output should be
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding
    out_height = int((height + np.sum(pad_h) - filter_height) / stride + 1)
    out_width = int((width + np.sum(pad_w) - filter_width) / stride + 1)

    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(channels), filter_height *
                  filter_width).reshape(-1, 1)
    return (k, i, j)


def determine_padding( filter_shape, output_shape="same"):

    # No padding
    if output_shape == "valid":
        return (0, 0), (0, 0)
    # Pad so that the output shape is the same as input shape (given that stride=1)
    elif output_shape == "same":
        filter_height, filter_width = filter_shape

        # Derived from:
        # output_height = (height + pad_h - filter_height) / stride + 1
        # In this case output_height = height and stride = 1. This gives the
        # expression for the padding below.
        pad_h1 = int(math.floor((filter_height - 1)/2))
        pad_h2 = int(math.ceil((filter_height - 1)/2))
        pad_w1 = int(math.floor((filter_width - 1)/2))
        pad_w2 = int(math.ceil((filter_width - 1)/2))

        return (pad_h1, pad_h2), (pad_w1, pad_w2)


def image_to_column( images, filter_shape, stride, output_shape='same'):

    filter_height, filter_width = filter_shape
    pad_h, pad_w = determine_padding(
        filter_shape, output_shape)  # Add padding to the image
    # Calculate the indices where the dot products are to be applied between weights
    images_padded = np.pad(
        images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')
    # and the image
    k, i, j = get_im2col_indices(
        images.shape, filter_shape, (pad_h, pad_w), stride)

    # Get content from image at those indices
    cols = images_padded[:, k, i, j]
    channels = images.shape[1]
    # Reshape content into column shape
    cols = cols.transpose(1, 2, 0).reshape(
        filter_height * filter_width * channels, -1)
    return cols



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
    def __init__(self, n_filters, filter_shape, padding='same', stride=1):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.trainable = True
        self.outputs = None
        self.derivated_inputs = None

    def initialize(self):
        # Initialize the weights
        filter_height, filter_width = self.filter_shape
        channels = self.input_shape[0]
        limit = 1 / math.sqrt(np.prod(self.filter_shape))
        self.W  = np.random.uniform(-limit, limit, size=(self.n_filters, channels, filter_height, filter_width))
        self.w0 = np.zeros((self.n_filters, 1))
        
    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward(self, X):
        self.input_shape = X.shape[1:]
        self.initialize()
        batch_size, channels, height, width = X.shape
        self.layer_input = X
        # Turn image shape into column shape
        # (enables dot product between input and weights)
        self.X_col = image_to_column(X, self.filter_shape, stride=self.stride, output_shape=self.padding)
        # Turn weights into column shape
        self.W_col = self.W.reshape((self.n_filters, -1))
        # Calculate output
        output = self.W_col.dot(self.X_col) + self.w0
        # Reshape into (n_filters, out_height, out_width, batch_size)
        output = output.reshape(self.output_shape() + (batch_size, ))
        # Redistribute axises so that batch size comes first
        output = output.transpose(3,0,1,2)
        self.outputs = output

    def backward(self, accum_grad):
        # Reshape accumulated gradient into column shape
        accum_grad = accum_grad.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)

        if self.trainable:
            # Take dot product between column shaped accum. gradient and column shape
            # layer input to determine the gradient at the layer with respect to layer weights
            grad_w = accum_grad.dot(self.X_col.T).reshape(self.W.shape)
            # The gradient with respect to bias terms is the sum similarly to in Dense layer
            grad_w0 = np.sum(accum_grad, axis=1, keepdims=True)

            # Update the layers weights
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        # Recalculate the gradient which will be propogated back to prev. layer
        accum_grad = self.W_col.T.dot(accum_grad)
        # Reshape from column shape to image shape
        accum_grad = column_to_image(accum_grad,
                                self.layer_input.shape,
                                self.filter_shape,
                                stride=self.stride,
                                output_shape=self.padding)

        self.derivated_inputs = accum_grad

    def output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride + 1
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride + 1
        return self.n_filters, int(output_height), int(output_width)

