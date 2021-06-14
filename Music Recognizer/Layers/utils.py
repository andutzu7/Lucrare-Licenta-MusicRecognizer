import numpy as np
import math

def column_to_image(images, images_shape, filter_shape, stride, output_shape='same'):
    """
        Determine_padding computes the necessary zero padding so the forward pass operation can take place.
    Args:   
        images(np.array): input batch of images.
        images_shape(tuple): The shape of the expected input of the layer. (batch_size, channels, height, width)
        filter_shape(tuple) : The filter that has to be applied over the input
        stride(int): The stride length of the filters during the convolution over the input.
        output_shape(string): Can have only 2 values, same and valid, for each output configuration (padded/non padded)

    Sources: * https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/layers.py
             * CS231n Stanford
    """
    # Extract the shape of the original image
    batch_size, channels, height, width = images_shape
    # Determine the padding in regards to the shape and filter
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    # Adjust the weight and height
    height_padded = height + np.sum(pad_h)
    width_padded = width + np.sum(pad_w)
    # Create an empty resied zero filled array 
    images_padded = np.zeros(
        (batch_size, channels, height_padded, width_padded))
    # Compute the indices where the dot products can be applied.
    k, i, j = get_im2col_indices(
        images_shape, filter_shape, (pad_h, pad_w), stride)
    # Compute the output
    images = images.reshape(channels * np.prod(filter_shape), -1, batch_size)
    images = images.transpose(2, 0, 1)
    # Add column content to the images at the indices
    np.add.at(images_padded, (slice(None), k, i, j),images)

    # Return image without padding
    return images_padded[:, :, pad_h[0]:height+pad_h[0], pad_w[0]:width+pad_w[0]]


def get_im2col_indices(images_shape, filter_shape, padding, stride=1):
    """
        Determine_padding computes the necessary zero padding so the forward pass operation can take place.
    Args:   
        images_shape(tuple): The shape of the expected input of the layer. (batch_size, channels, height, width)
        filter_shape(tuple) : The filter that has to be applied over the input
        stride(int): The stride length of the filters during the convolution over the input.
        stride(int):The stride length of the filters during the convolution over the input.

    Sources: * https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/layers.py
             * CS231n Stanford
    """
   
    # Extract the shape of the original image
    batch_size, channels, height, width = images_shape
    # Extract the height and width of the filter
    filter_height, filter_width = filter_shape
    # Extract the padding width and height
    pad_h, pad_w = padding
    # Compute the output dimensions
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


def determine_padding(filter_shape, output_shape="same"):
    """
        Determine_padding computes the necessary zero padding so the forward pass operation can take place.
    Args:   
        filter_shape(tuple) : The filter that has to be applied over the input
        output_shape(string): can have only 2 values, same and valid, for each output configuration (padded/non padded)

    Sources: * https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/layers.py
             * CS231n Stanford
    """
    # If the output_shape is set to "valid" then no padding is performed
    if output_shape == "valid":
        return (0, 0), (0, 0)

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



def image_to_column(images, filter_shape, stride, output_shape='same'):
    """
        Image to column flattens the image input then extracts the information from it regarding
        the filter_shape.
    Args:   
        images(np.array) : Input image
        filter_shape(tuple) : The filter that has to be applied over the input
        stride(int): The stride length of the filters during the convolution over the input.
        output_shape(string): can have only 2 values, same and valid, for each output configuration (padded/non padded)

    Sources: * https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/layers.py
             * CS231n Stanford
    """
    filter_height, filter_width = filter_shape
    # Determine the image padding
    pad_h, pad_w = determine_padding(
        filter_shape, output_shape)  
    # Pad the image
    images_padded = np.pad(
        images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')
    # Compute the indices where the dot products can be applied.
    k, i, j = get_im2col_indices(
        images.shape, filter_shape, (pad_h, pad_w), stride)
    # Extract the data from image using the computed indices
    output = images_padded[:, k, i, j]
    # Extract the channels of the input image
    channels = images.shape[1]
    # Reshape the output to column shape 
    output = output.transpose(1, 2, 0).reshape(
        filter_height * filter_width * channels, -1)
    return output