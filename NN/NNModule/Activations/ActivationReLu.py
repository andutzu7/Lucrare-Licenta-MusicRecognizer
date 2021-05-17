import numpy as np
# ReLU activation
class ActivationReLu:
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)
    # Backward pass
    def backward(self, derivated_values):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.derivated_inputs = derivated_values.copy()
        # Zero gradient where input values were negative
        self.derivated_inputs[self.inputs <= 0] = 0
    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs