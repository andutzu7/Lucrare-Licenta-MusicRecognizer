import numpy as np
from .Loss import Loss

class MeanSquaredError(Loss):  
    """
        The MSE class computes the loss by calculating the average of the squares of the errors
        (the average squared difference between the estimated and actual values)

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.426-430]

    """
    def forward(self, y_pred, y_true):
        """
        Performs the forward pass. 

        Args :  y_pred(np.array): Model predictions       
                y_true(np.array): Actual values

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.426-430]
        """
        # Perform the MSE loss.
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses

    def backward(self, derivated_values, y_true):
        """
        Perform the backward pass.

        Args : derivated_values(np.array): Input values.
               y_true(np.array): Actual values

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.426-430]
        """
        # Taking the number of outputs in every sample we'll use 
        outputs = len(derivated_values[0])
        # Applying the gradient on the values
        self.derivated_inputs = -2 * (y_true - derivated_values) / outputs