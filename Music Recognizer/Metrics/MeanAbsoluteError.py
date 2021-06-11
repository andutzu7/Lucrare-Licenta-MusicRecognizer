import numpy as np
from .Loss import Loss

# Mean Absolute Error loss


class MeanAbsoluteError(Loss):  # L1 loss
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        # Return losses
        return sample_losses
    # Backward pass

    def backward(self, derivated_values, y_true):
        # Number of samples
        samples = len(derivated_values)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(derivated_values[0])
        # Calculate gradient
        self.derivated_inputs = np.sign(y_true - derivated_values) / outputs
        # Normalize gradient
        self.derivated_inputs = self.derivated_inputs / samples
