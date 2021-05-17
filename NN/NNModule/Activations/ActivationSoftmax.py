import numpy as np
# Softmax activation
class ActivationSoftmax:
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        self.output = probabilities
    # Backward pass
    def backward(self, derivated_values):
        # Create uninitialized array
        self.derivated_inputs = np.empty_like(derivated_values)
        # Enumerate outputs and gradients
        for index, (single_output, single_derivated_values) in \
                enumerate(zip(self.output, derivated_values)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.derivated_inputs[index] = np.dot(jacobian_matrix,
                                         single_derivated_values)
    # Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)