import numpy as np
import nnfs
import os
import cv2
import pickle
import copy
nnfs.init()

# Layer initialization


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        # self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.weights = np.array([[0.01764052,  0.00400157,  0.00978738,  0.02240893,  0.01867558],
                        [-0.00977278, 0.00950088, -0.00151357, -0.00103219,  0.00410599],
                        [0.00144044,  0.01454273,  0.00761038,0.00121675,  0.00443863],
                        [0.00333674,  0.01494079, -0.00205158,0.00313068, -0.00854096],
                        [-0.0255299,  0.00653619,  0.00864436, -0.00742165,  0.02269755]])
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    # Forward pass

    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs  # Dense layer
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    # Backward pass

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        self.weight_regularizer_l2 = 0.003
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                self.biases
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
    # Retrieve layer parameters

    def get_parameters(self):
        return self.weights, self.biases
    # Set weights and biases in a layer instance

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
# Dropout


class Layer_Dropout:
    # Init
    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate
    # Forward pass

    def forward(self, inputs, training):
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

    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask
# Input "layer"


class Layer_Input:
    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs


if __name__ == "__main__":
    dl = Layer_Dropout(0.03)

    inputs = np.array([[0.01764052,  0.00400157,  0.00978738,  0.02240893,  0.01867558],
              [-0.00977278, 0.00950088, -0.00151357, -0.00103219,  0.00410599],
              [0.00144044,  0.01454273,  0.00761038,  0.00121675,  0.00443863],
              [0.00333674,  0.01494079, -0.00205158,  0.00313068, -0.00854096],
              [-0.0255299,  0.00653619,  0.00864436, -0.00742165,  0.02269755]])

    dl.forward(inputs,True)
    dl.backward(inputs)
    print(dl.inputs)
    print(dl.binary_mask)
    print(dl.output)
    print(dl.dinputs)
    #print(dl.inputs,dl.binary_mask,dl.output,dl.dinputs)
