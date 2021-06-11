import numpy as np
# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    # Backward pass
    def backward(self, derivated_values, y_true):
        # Number of samples
        samples = len(derivated_values)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.derivated_inputs = derivated_values.copy()
        # Calculate gradient
        self.derivated_inputs[range(samples), y_true] -= 1