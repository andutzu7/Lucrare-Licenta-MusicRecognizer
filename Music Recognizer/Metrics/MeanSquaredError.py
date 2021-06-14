import numpy as np
from .Loss import Loss

class MeanSquaredError(Loss):  
    """
        The MSE class computes the loss by calculating the average of the squares of the errors
        (the average squared difference between the estimated and actual values)


        Each object that inherits this class must implement the 'forward' method.

    The Loss class contains:
        :param weights(np.array) : Dense Layer's weights.
        :param biases(np.array) : Dense Layer's biases
        :param weight_regularizer_l1(float) : l1 weight regulizer factor
        :param weight_regularizer_l2(float) : l2 weight regulizer factor
        :param bias_regularizer_l1(float) : l1 bias regulizer factor
        :param bias_regularizer_l2(float) : l2 bias regulizer factor

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.129-139]

    """
    # Forward pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        # Return losses
        return sample_losses
    # Backward pass
    def backward(self, derivated_values, y_true):
        # Number of samples
        samples = len(derivated_values)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(derivated_values[0])
        # Gradient on values
        self.derivated_inputs = -2 * (y_true - derivated_values) / outputs