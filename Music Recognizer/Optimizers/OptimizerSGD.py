import numpy as np

class OptimizerSGD:
    """
        The SGD (Stochastic Gradient Descent) creates a rolling average of the gradients over some
        number of updates and uses the average with the unique gradient at each step.

        The SGD class has the following parameters:
        
        :param learning_rate(float): the optimizer learning rate after the forward pass.
        :param current learning rate(float): the previous iteration learning rate.
        :param decay(float): decay rate.
        :param momentum(float): a constant rate that helps the model avoid the local minimum.
        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.283-290]

    """
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        """                         
        The function calculates the current learning rate if the decay is set, in regards to the
        current iteration.

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.283-290]
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Updates the parameters.

        
        Args:
        :param layer(layer type): layer parameter.

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.283-290]
        """
        # If the momentum is defined
        if self.momentum:
            # If layer does not contain momentum arrays, create them filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            # Build weight updates with momentum - take previous updates multiplied by 
            # the retain factor and update it with current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.derivated_weights
            layer.weight_momentums = weight_updates

            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.derivated_biases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * \
                layer.derivated_weights
            bias_updates = -self.current_learning_rate * \
                layer.derivated_biases
        # Update weights and biases using either vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        """
        Increment the number of iterations before each update.

        Args:

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.283-290]
        """
        self.iterations += 1
