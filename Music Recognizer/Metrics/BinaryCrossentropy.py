import numpy as np
from .Loss import Loss

class BinaryCrossentropy(Loss):
    """
        The class computes the binary crossentropy by applying the formula.

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.407-412]

    """
    def forward(self, y_pred, y_true):
        """
        Performs the forward pass. 

        Args :  y_pred(np.array): Model predictions       
                y_true(np.array): Actual values

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.112-116]
        """
        # Clip data to prevent division by 0 and to avoid draging the mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Calculate the loss by sample
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        # Compute the mean
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses

    def backward(self, derivated_values, y_true):
        """
        Perform the backward pass.

        Args : derivated_values(np.array): Input values.
               y_true(np.array): Actual values

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.430-436]
        """
        # Number of samples and outputs
        samples = len(derivated_values)
        outputs = len(derivated_values[0])
        # Clip data to prevent division by 0 and to avoid draging the mean towards any value
        clipped_derivated_values = np.clip(derivated_values, 1e-7, 1 - 1e-7)
        # Calculate gradient
        self.derivated_inputs = -(y_true / clipped_derivated_values -
                         (1 - y_true) / (1 - clipped_derivated_values)) / outputs
        # Normalize the gradient and applying it to the values
        self.derivated_inputs = self.derivated_inputs / samples
