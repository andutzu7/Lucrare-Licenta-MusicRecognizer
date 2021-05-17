import numpy as np
from .Loss import Loss

# Binary cross-entropy loss
class BinaryCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        # Return losses
        return sample_losses
    # Backward pass

    def backward(self, derivated_values, y_true):
        # Number of samples
        samples = len(derivated_values)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(derivated_values[0])
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_derivated_values = np.clip(derivated_values, 1e-7, 1 - 1e-7)
        # Calculate gradient
        self.derivated_inputs = -(y_true / clipped_derivated_values -
                         (1 - y_true) / (1 - clipped_derivated_values)) / outputs
        # Normalize gradient
        self.derivated_inputs = self.derivated_inputs / samples
