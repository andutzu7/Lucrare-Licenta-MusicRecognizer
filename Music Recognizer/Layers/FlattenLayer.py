import numpy as np

class Flatten:
    """
    The Flatten array turns the given input to a 1D array
        
    The Flatten class contains:
        :param prev_shape(tuple): The initial input shape (necessary for the backpropagation).

    Sources: *https://nickmccullum.com/python-deep-learning/flattening-full-connection/ 
             *https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/layers.py
    """
    def __init__(self):
        self.prev_shape = None
        self.derivated_inputs = None
        self.output = None

    def forward(self, inputs):
        """
        Performs the forward pass by reshaping the array into 1D flattened form.
        Args:
            inputs (np.array): given inputs.

        Sources:    
             *https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/layers.py
        """
        # Eliminating the batch size
        self.input_shape = inputs.shape[1:]
        # Storing the previous shape
        self.prev_shape = inputs.shape
        # Flattening the array
        self.output = inputs.reshape((inputs.shape[0], -1))

    def backward(self, derivated_values):
        """
        Performs the backward pass by reshaping the derivated values to the initial 
        dimensions.

        Args:
            derivated_values (np.array): Derivated inputs .

        Sources: 
             *https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/layers.py
        """
        self.derivated_inputs = derivated_values.reshape(self.prev_shape)