import numpy as np
"""
The Softmax function handles the inputs by producing a normalized distribution of probabilities over the given input data.
The formula for the Softmax activation function is: S_{i,j} = e^{z_{i,l}} / sum(e^{z_{i,l}}) where
                                                            i is the indices of the current sample
                                                            j is the indices of the current output value
                                                            z is a given output


The Softmax class contains:
    :param inputs(np.array) : A numpy array containing the output of the layer the function is attached to
    :param output(np.array) : The 'clipped' array, after the activation function has been applied.
    :param derivated_inputs(np.array) : The outputs of the class after the backward pass
   
Sources: * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.98-99]
"""
class ActivationSoftmax:
    def forward(self, inputs):
        """
        Performs the forward pass by applying the formula.
        Args:
            inputs(np.array): Previous layer's outputs.
            training(boolean): Specifying if the network is in training or inference mode.
        Sources: * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.104]
        """
        # Saving the input values
        self.inputs = inputs
        # Performing the exponentiation of the input values
        # The exponentiation is performed column wise (axis=1) so the function can be applied to the batches of data
        # For avoiding dead neurons and very large numbers that can cause explosions(integer overflow), we substract
        # the max value out of each input batch
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize the values for each sample by applying the Softmax's formula
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        self.output = probabilities
    def backward(self, derivated_values):
        """
        Performs the backward pass using the gradient chaining method.

        Args:
            derivated_values(np.array): Derivated inputs array.
        Sources: * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.226-229]
        """
        # Creating an array with the same shape as the inputs
        self.derivated_inputs = np.empty_like(derivated_values)
        # Iterate through the outputs and the derivated_values(gradients)
        for index, (single_output, single_derivated_values) in \
                enumerate(zip(self.output, derivated_values)):
            # Flatten the output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix (the matrix of current sample's partial derivative)
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # We used both single_output and jacobian_matrix in a flatten form in order to avoid the output being a 3D array.
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.derivated_inputs[index] = np.dot(jacobian_matrix,
                                         single_derivated_values)
    def predictions(self, outputs):
        """
        Calculate the output's predictions by returning the index of the maximal value row wise.

        Args:
            outputs(np.array): Input values.
        """
        return np.argmax(outputs, axis=1)