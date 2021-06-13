import numpy as np

"""
The Sigmoid function handles the inputs using the by activating them using the formula 1/1+e^(-x), reducing the range
of the input value(x) to the interval [0,1]. The Sigmoid function is used especially in binary classification problems.

The ActivationSigmoid class contains:
    :param inputs(np.array) : A numpy array containing the output of the layer the function is attached to
    :param output(np.array) : The 'clipped' array, after the activation function has been applied.
    :param derivated_inputs(np.array) : The outputs of the class after the backward pass
   
Sources: * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.75-76]
         * https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
"""
class Activation_Sigmoid:
    def forward(self, inputs,training=True):
        """
        Performs the forward pass by mapping the inputs to the [0,1] interval.

        Args:
            inputs(np.array): Previous layer's outputs.
            training (boolean): Specifying if the network is in training or inference mode.
        """
        # Saving the input values
        self.inputs = inputs
        # Computing the outputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, derivated_values):
        """
        Performs the backward pass using the gradient chaining method.

        Args:
            derivated_values(np.array): Derivated inputs array.
        """
        # Calculating the outputs by applying the function's derivative.
        self.derivated_inputs = derivated_values * (1 - self.output) * self.output

    def predictions(self, outputs):
        """
        Calculate the output's predictions. The sigmoid layer predictions are computed by doing
        a comparasing with the interval's middle threshold (0.5) and returning the respective value.

        Args:
            outputs(np.array): Input values.

        Source: * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.486]
        """
        # Computing each individual value of the array by checking whether or not its value its above 
        # the threshold, reconverting it to an integer value afterwards
        return (outputs > 0.5) * 1
