import numpy as np
import typing

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

    def forward(self, inputs: np.array, epsilon: np.double = 0.001, momentum: np.double = 0.999, training: bool = True) -> None:
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


        """

        cache = None
        if len(inputs.shape) == 2:
            _, D = inputs.shape

            if self.moving_mean is None and self.moving_variation is None:
                self.moving_mean = np.zeros(D, dtype=inputs.dtype)
                self.moving_variation = np.ones(D, dtype=inputs.dtype)

            if self.gamma is None and self.beta is None:
                self.gamma = np.ones(D, dtype=inputs.dtype)
                self.beta = np.zeros(D, dtype=inputs.dtype)

        elif len(inputs.shape) == 4:

            N, C, H, W = inputs.shape

            if self.moving_mean is None and self.moving_variation is None:
                self.moving_mean = np.zeros((1, C, 1, 1), dtype=inputs.dtype)
                self.moving_variation = np.ones(
                    (1, C, 1, 1), dtype=inputs.dtype)

            if self.gamma is None and self.beta is None:
                self.gamma = np.ones((1, C, 1, 1), dtype=inputs.dtype)
                self.beta = np.zeros((1, C, 1, 1), dtype=inputs.dtype)

        moving_mean = self.moving_mean
        moving_variation = self.moving_variation

        gamma = self.gamma
        beta = self.beta

        # gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta
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

            cache = (inputs_norm, standard_deviation, gamma)

        elif training == False:
            inputs_norm = (inputs - moving_mean) / \
                np.sqrt(moving_variation + epsilon)
            out = gamma * inputs_norm + beta

        self.moving_mean = moving_mean
        self.moving_variation = moving_variation
        self.cache = cache
        self.output = out

    def backward(self, derivated_values):
        """
        Performs the backward pass using the gradient chaining method.

        Args:
            derivated_values (np.array): Derivated inputs minibatch.

        """
        if len(derivated_values.shape) == 2:
            N = derivated_values.shape[0]
            inputs_norm, std, gamma = self.cache

            dinputs_norm = derivated_values * gamma
            dinputs = 1 / N / std * (N * dinputs_norm -
                                     dinputs_norm.sum(ainputsis=0) -
                                     inputs_norm * (dinputs_norm * inputs_norm).sum(axis=0))
        elif len(derivated_values.shape) == 4:

            N, C, H, W = derivated_values.shape
            inputs_norm, std, gamma = self.cache

            dinputs_norm = derivated_values * gamma
            dinputs = 1 / C / std * (C * dinputs_norm -
                                     dinputs_norm.sum(axis=(0, 2, 3)) -
                                     inputs_norm * (dinputs_norm * inputs_norm).sum(axis=(0, 2, 3)))

        self.derivated_inputs = dinputs 
