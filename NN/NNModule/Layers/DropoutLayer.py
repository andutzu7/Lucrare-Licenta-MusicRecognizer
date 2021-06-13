import numpy as np
# Dropout


class DropoutLayer:
    # Init
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate
    # Forward pass

    # de schimbat cu true
    def forward(self, inputs, training=True):
        # Save input values
        self.inputs = inputs
        # If not in the training mode - return values

        if not training:
            self.output = inputs.copy()
            return
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate,
                                              size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask
    # Backward pass

    def backward(self, derivated_values):
        # Gradient on values
        self.derivated_inputs = derivated_values * self.binary_mask
