import numpy as np


class DenseLayer:
    """
        DenseLayer/FullyConectedLayer adds a non-linearity propriety to the network and performs the
        learning by multiplying the inputs to the weights, then adding the biases.

    The DenseLayer class contains:
        :param weights(np.array) : Dense Layer's weights.
        :param biases(np.array) : Dense Layer's biases
        :param output(np.array) : The outputs of the layer after the forward pass
        :param weight_regularizer_l1(float) : l1 weight regulizer factor
        :param weight_regularizer_l2(float) : l2 weight regulizer factor
        :param bias_regularizer_l1(float) : l1 bias regulizer factor
        :param bias_regularizer_l2(float) : l2 bias regulizer factor
        :param inputs(np.array) : inputs array
        :param derivated_inputs(np.array) : The outputs of the layer after the backward pass

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.66-71]

    """
    def __init__(self, inputs_number, neuron_number,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(inputs_number, neuron_number)
        self.biases = np.zeros((1, neuron_number))
        self.inputs = None
        self.output = None
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        """
        Performs the forward pass by multipyling the weights with the inputs then adding the biases.
        Args:
            inputs (np.array): given inputs.

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.66-71]
        """
        # Storing the input values
        self.inputs = inputs
        # Apply the formula
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, derivated_values):
        """
        Performs the backward pass by using the gradient chaining method.

        Args:
            derivated_values (np.array): Derivated inputs .

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.66-71]
        
        """
        # Computing the derivated values
        self.derivated_weights = np.dot(self.inputs.T, derivated_values)
        self.derivated_biases = np.sum(derivated_values, axis=0, keepdims=True)
        # Applying gradients in regards to regularisation if they re initialized
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.derivated_weights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.derivated_weights += 2 * self.weight_regularizer_l2 * \
                             self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.derivated_biases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.derivated_biases += 2 * self.bias_regularizer_l2 * \
                            self.biases
        # Computing the output
        self.derivated_inputs = np.dot(derivated_values, self.weights.T)

    def get_parameters(self):
        """
        Return the layer's current parameters. 
        
        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.66-71]
        """
        return self.weights, self.biases