import numpy as np


class DropoutLayer:
    """
    DropoutLayer discards a given rate of the inputs neurons.
        
    The DropoutLayer class contains:
        :param rate(float): The dropout rate
        of the output shape.
        :param binary_mask(np.array) : Mask to be applied over the inputs.
        :param inputs(np.array) : inputs array
        :param output(np.array) : The outputs of the layer after the forward pass

    Sources:  * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.361-371]

    """
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training=True):
        """
        Performs the forward pass by multipyling the weights with the inputs then adding the biases.
        Args:
            inputs (np.array): given inputs.
            training (bool): Value that specifies the behaviour of the layer.

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.66-71]
        """
        # Save input values
        self.inputs = inputs

        # If training is set to false, return the values
        if not training:
            self.output = inputs.copy()
            return
        # Else generate a binary mask to deactivate neurons at the given rate
        self.binary_mask = np.random.binomial(1, self.rate,
                                              size=inputs.shape) / self.rate
        # Apply mask to the output values
        self.output = inputs * self.binary_mask

    def backward(self, derivated_values):
        """
        Performs the backward pass by using the gradient chaining method.

        Args:
            derivated_values (np.array): Derivated inputs .

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.66-71]
        
        """
        # Applying the gradient on values
        self.derivated_inputs = derivated_values * self.binary_mask
