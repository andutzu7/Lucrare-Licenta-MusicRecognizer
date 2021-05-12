import numpy as np
# ReLU activation


class Activation_Softmax:
    # Forward pass
    def forward(self, inputs, training):
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

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                             np.dot(single_output, single_output.T)
            # # Calculate sample-wise gradient
            # # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)
        # Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
# Sigmoid activation
if __name__ == "__main__":
    rl=Activation_Softmax()

    inputs=np.array([[0.01764052,  0.00400157,  0.00978738,  0.02240893,  0.01867558],
              [-0.00977278, 0.00950088, -0.00151357, -0.00103219,  0.00410599],
              [0.00144044,  0.01454273,  0.00761038,  0.00121675,  0.00443863],
              [0.00333674,  0.01494079, -0.00205158,  0.00313068, -0.00854096],
              [-0.0255299,  0.00653619,  0.00864436, -0.00742165,  0.02269755]])

    rl.forward(inputs, True)
    rl.backward(inputs)
    print(rl.output)
    print(rl.dinputs)
