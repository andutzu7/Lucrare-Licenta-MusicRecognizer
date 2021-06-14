import numpy as np

class Activation_Softmax_Loss_CategoricalCrossentropy:
    """
    Combined Softmax activation and cross-entropy loss for faster backward chaining.
    """

    def backward(self, derivated_values, y_true):
        """
        Perform the backward pass.

        Args : derivated_values(np.array): Input values.
               y_true(np.array): Actual values

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.430-436]
        """
        samples = len(derivated_values)
        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.derivated_inputs = derivated_values.copy()
        # Calculate gradient
        self.derivated_inputs[range(samples), y_true] -= 1