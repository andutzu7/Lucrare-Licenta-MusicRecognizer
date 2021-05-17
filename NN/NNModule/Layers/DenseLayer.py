import numpy as np

# De facut un docstring misto despre clasa in care sa scriu si toate fieldurile pe care le are

class DenseLayer:
    # Layer initialization
    def __init__(self, inputs_number, neuron_number,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(inputs_number, neuron_number)
        self.biases = np.zeros((1, neuron_number))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    # Backward pass

    def backward(self, derivated_values):
        # Gradients on parameters
        self.derivated_weights = np.dot(self.inputs.T, derivated_values)
        self.derivated_biases = np.sum(derivated_values, axis=0, keepdims=True)
        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.derivated_weights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.derivated_weights += 2 * self.weight_regularizer_l2 * \
                             self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.derivated_biases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.derivated_biases += 2 * self.bias_regularizer_l2 * \
                            self.biases
        # Gradient on values
        self.derivated_inputs = np.dot(derivated_values, self.weights.T)
    # Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases
    # Set weights and biases in a layer instance
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases