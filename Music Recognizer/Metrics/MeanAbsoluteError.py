import numpy as np
from .Loss import Loss

class MeanAbsoluteError(Loss):  
    """
        The MAE class computes the loss by calculating the average of the absolute value of the errors
        (the average squared difference between the estimated and actual values)

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.430-436]

    """
    def forward(self, y_pred, y_true):
        """
        Performs the forward pass. 

        Args :  y_pred(np.array): Model predictions       
                y_true(np.array): Actual values

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.430-436]
        """
        # Perform the MAE error.
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses

    def backward(self, derivated_values, y_true):
        """
        Perform the backward pass.

        Args : derivated_values(np.array): Input values.
               y_true(np.array): Actual values

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.430-436]
        """
        # Taking the number of outputs in every sample we'll use 
        samples = len(derivated_values)
        outputs = len(derivated_values[0])
        # Calculate the gradient
        self.derivated_inputs = np.sign(y_true - derivated_values) / outputs
        # Normalize the gradient and applying it to the values
        self.derivated_inputs = self.derivated_inputs / samples
