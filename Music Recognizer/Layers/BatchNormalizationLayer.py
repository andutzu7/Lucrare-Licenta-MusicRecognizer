import numpy as np

"""
BatchNormalization is a technique for standardizing the inputs of a mini-batch,
with the effect of stabilizing the learning process and dramatically reducing 
the number of training epochs required.
The BatchNormalization model works by applying a linear scale and then shift the 
result to the minibatch.

The BatchNormalization class contains:
    :param gamma(np.array) : Batch Normalization's scaling factor.
    :param beta(np.array) : Batch Normalization's offset factor
    :param moving_mean(np.array) : The minibatch's mean(\mu)
    :param moving_variation(np.array) :  The minibatch's mean (\tau)
    :param cache(tuple) : A tuple containing the necessary parameters for the backpropagation.
    :param output(np.array) : The outputs of the layer after the forward pass
    :param derivated_inputs(np.array) : The outputs of the layer after the backward pass

    Sources:    *https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/
                *https://towardsdatascience.com/batch-normalisation-in-deep-neural-network-ce65dd9e8dbf
"""


class BatchNormalization:

    def __init__(self):
        self.gamma = None
        self.beta = None
        self.moving_mean = None
        self.moving_variation = None
        self.cache = None
        self.output = None
        self.derivated_inputs = None

    def forward(self, inputs, epsilon=0.001, momentum=0.999, training=True):
        """
        Performs the forward pass. 
        First it checks the number of dimensions of the inputs, handling either cases(2d dense layers and 4d convolutional layers)
        accordingly in order to compute the moving mean(the mean of the sum of the input minibatch values) and moving variation
        (the mean of the squared sum of the difference between the input minibatch values and the moving mean).

        It then computes the outputs using the formula
        gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta
        if training is set to True
        or gamma * (batch - self.moving_mean) / sqrt(self.moving_variation + epsilon) + beta otherwise
        where:
            gamma(np.array) : Batch Normalization's scaling factor.
            batch(np.array) : A minibatch of data
            beta(np.array) : Batch Normalization's offset factor
            moving_mean(np.array) : The minibatch's mean(\mu)
            moving_variation(np.array) :  The minibatch's mean (\tau)

        Args:
            inputs (np.array): Inputs minibatch.
            epsilon=0.001 (double): A constant that prevents the division by 0
            momentum=0.999 (double): A constant for setting the moving mean's momentum
            training=True (boolean): Specifying if the network is in training or inference mode.

        Sources:    *https://gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-batch-norm-scratch.html
                    *https://analyticsindiamag.com/hands-on-guide-to-implement-batch-normalization-in-deep-learning-models/
                    *https://github.com/eriklindernoren/ML-From-Scratch#machine-learning-from-scratch
        """

        cache = None
        # Handling the case where the previous layer is a dense layer (2D input array)
        if len(inputs.shape) == 2:
            # Extracting the input dimensions from the input
            _, D = inputs.shape
            # Initializing the moving mean and moving variation for the first iteration (the only time where they can be
            #  None during the execution)
            if self.moving_mean is None and self.moving_variation is None:
                # The moving mean will have D dimensions and the inputs data type
                self.moving_mean = np.zeros(D, dtype=inputs.dtype)
                # Same for the moving variation
                self.moving_variation = np.ones(D, dtype=inputs.dtype)

            # Initializing the gamma(scaling factor) and beta(offset factor) for the first iteration (the only time where they can be
            #  None during the execution)
            if self.gamma is None and self.beta is None:
                # Gamma will have D dimensions and the inputs data type
                self.gamma = np.ones(D, dtype=inputs.dtype)
                # Beta will have D dimensions and the inputs data type
                self.beta = np.zeros(D, dtype=inputs.dtype)

        # Handling the case where the previous layer is a (2D input array)
        elif len(inputs.shape) == 4:
            # Exctracting the dimensions of the inputs
            N, C, H, W = inputs.shape

            # Initializing the moving mean and moving variation for the first iteration (the only time where they can be
            # None during the execution)
            if self.moving_mean is None and self.moving_variation is None:
                # The moving mean will be a 4d array with the same number of channels as the input
                self.moving_mean = np.zeros((1, C, 1, 1), dtype=inputs.dtype)
                # The moving_variation mean will be a 4d array with the same number of channels as the input
                self.moving_variation = np.ones(
                    (1, C, 1, 1), dtype=inputs.dtype)

            # Initializing the gamma(scaling factor) and beta(offset factor) for the first iteration (the only time where they can be
            # None during the execution)
            if self.gamma is None and self.beta is None:
                # Gamma will be a 4d array with the same number of channels as the input
                self.gamma = np.ones((1, C, 1, 1), dtype=inputs.dtype)
                # Beta will be a 4d array with the same number of channels as the input
                self.beta = np.zeros((1, C, 1, 1), dtype=inputs.dtype)

        # Initializing the local moving mean and moving variations with the values from the layer
        moving_mean = self.moving_mean
        moving_variation = self.moving_variation

        # Initializing the local gamma and beta with those with the values from the layer
        gamma = self.gamma
        beta = self.beta

        '''
        Applying the formula gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta by the following steps:
            *Handling the 2d and 4d cases separately 
            *Computing the mean of the inputs along rows 
            *Computing the variance of the inputs along rows
            *Computing the moving mean 
            *Computing the moving variation
            *Computing the standard deviation
            *Centering the inputs
            *Normalizing the inputs
            *Calculating the output values
        '''
        if training:
            if len(inputs.shape) == 2:
                sample_mean = inputs.mean(axis=0)
                sample_var = inputs.var(axis=0)

                moving_mean = momentum * moving_mean + \
                    (1 - momentum) * sample_mean
                moving_variation = momentum * \
                    moving_variation + (1 - momentum) * sample_var
                standard_deviation = np.sqrt(sample_var + epsilon)
                inputs_centered = inputs - sample_mean
                inputs_norm = inputs_centered / standard_deviation
                out = gamma * inputs_norm + beta
            elif len(inputs.shape) == 4:

                sample_mean = inputs.mean(axis=(0, 2, 3))
                sample_var = inputs.var(axis=(0, 2, 3))

                moving_mean = momentum * moving_mean + \
                    (1 - momentum) * sample_mean.reshape((1, C, 1, 1))
                moving_variation = momentum * \
                    moving_variation + (1 - momentum) * \
                    sample_var.reshape((1, C, 1, 1))

                standard_deviation = np.sqrt(
                    sample_var.reshape((1, C, 1, 1)) + epsilon)
                inputs_centered = inputs - sample_mean.reshape((1, C, 1, 1))
                inputs_norm = inputs_centered / standard_deviation
                out = gamma * inputs_norm + beta
            # Creating the cache tuple that contains necessary values for the backpropagation
            cache = (inputs_norm, standard_deviation, gamma)
        # Applying the formula gamma * (batch - self.moving_mean) / sqrt(self.moving_variation + epsilon) + beta 
        elif training == False:
            inputs_norm = (inputs - moving_mean) / \
                np.sqrt(moving_variation + epsilon)
            out = gamma * inputs_norm + beta

        # Storing the newly calculated values in the layer
        self.moving_mean = moving_mean
        self.moving_variation = moving_variation
        self.cache = cache
        self.output = out

    def backward(self, derivated_values):
        """
        Performs the backward pass using the gradient chaining method.

        Args:
            derivated_values (np.array): Derivated inputs minibatch.

        Sources: *https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
                 *https://kevinzakka.github.io/2016/09/14/batch_normalization/   
                 *https://github.com/eriklindernoren/ML-From-Scratch#machine-learning-from-scratch
        """
        if len(derivated_values.shape) == 2:
            N = derivated_values.shape[0]
            inputs_norm, std, gamma = self.cache

            dinputs_norm = derivated_values * gamma
            dinputs = 1 / N / std * (N * dinputs_norm -
                                     dinputs_norm.sum(axis=0) -
                                     inputs_norm * (dinputs_norm * inputs_norm).sum(axis=0))
        elif len(derivated_values.shape) == 4:

            N, C, H, W = derivated_values.shape
            inputs_norm, std, gamma = self.cache

            dinputs_norm = derivated_values * gamma
            dinputs = 1 / C / std * (C * dinputs_norm -
                                     dinputs_norm.sum(axis=(0, 2, 3)) -
                                     inputs_norm * (dinputs_norm * inputs_norm).sum(axis=(0, 2, 3)))

        self.derivated_inputs = dinputs
