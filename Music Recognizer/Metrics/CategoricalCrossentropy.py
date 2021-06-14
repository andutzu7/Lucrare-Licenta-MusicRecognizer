import numpy as np
from .Loss import Loss


class CategoricalCrossentropy(Loss):
    """
        The class computes the categorical crossentropy by applying the formula: 
            -sum(y_{i,j}*log(y_hat_{i,j}))

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.112-116]

    """
    def forward(self, y_pred, y_true):
        """
        Performs the forward pass. 

        Args :  y_pred(np.array): Model predictions       
                y_true(np.array): Actual values

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.112-116]
        """
        samples = len(y_pred)
        # Clip data to prevent division by 0 and to avoid draging the mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values 
        # If the shape of the input is of the categorical form
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # If the shape of the input is of the one-hot encoding label
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        # Compute the loss
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, derivated_values, y_true):
        """
        Perform the backward pass.

        Args : derivated_values(np.array): Input values.
               y_true(np.array): Actual values

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.430-436]
        """
        samples = len(derivated_values)
        labels = len(derivated_values[0])
        # If the shape of the input is of the categorical form
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate the gradient
        self.derivated_inputs = -y_true / derivated_values
        # Normalize the gradient and applying it to the values
        self.derivated_inputs = self.derivated_inputs / samples
