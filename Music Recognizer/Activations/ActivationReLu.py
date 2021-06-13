import numpy as np
"""
ReLU(Rectified Linear Units Activation Function) is a non-linear function that handles the inputs
by activating them using the formula max(x,0).
ReLU is a very popular choice because of its ability to modelate linear problems into nonlinears ones
and because of its simple computing formula.

The ActivationReLu class contains:
    :param inputs(np.array) : A numpy array containing the output of the layer the function is attached to
    :param output(np.array) : The 'clipped' array, after the activation function has been applied.
    :param derivated_inputs(np.array) : The outputs of the class after the backward pass
   
Sources: * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.76-77]
         * https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
         * https://medium.com/analytics-vidhya/how-relu-works-f317a947bdc6
"""
class ActivationReLu:
    def forward(self, inputs):
        """
        Performs the forward pass by 'clipping' the array of the values below 0. 

        Args:
            inputs(np.array): Previous layer's outputs.
            training(boolean): Specifying if the network is in training or inference mode.
        """
        # Saving the input values
        self.inputs = inputs
        # Computing the outputs after applying the maximum function (returning an array containing the element wise maximum)
        self.output = np.maximum(0, inputs)

    def backward(self, derivated_values):
        """
        Performs the backward pass using the gradient chaining method.

        Args:
            derivated_values(np.array): Derivated inputs array.
        """
        # Copying the input values so we don't modyfy the reference to the data
        self.derivated_inputs = derivated_values.copy()
        # Apply the zero gradient to the values.
        self.derivated_inputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        """
        Calculate the output's predictions. Since relu falls into the Linear Activation functions, the prediction
        will simply be the outputs.

        Args:
            outputs(np.array): Input values.
        """
        return outputs