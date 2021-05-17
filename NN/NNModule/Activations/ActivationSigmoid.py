import numpy as np
# Sigmoid activation

class Activation_Sigmoid:
    # Forward pass
    def forward(self, inputs):
        # Save input and calculate/save output
        # of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        # Backward pass

    def backward(self, derivated_values):
        # Derivative - calculates from output of the sigmoid function
        self.derivated_inputs = derivated_values * (1 - self.output) * self.output
    # Calculate predictions for outputs

    def predictions(self, outputs):
        return (outputs > 0.5) * 1
