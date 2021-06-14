import numpy as np
# RMSprop optimizer


class OptimizerRMSprop:
    """
        RMSProp (Root Mean Square Propagation) calculates an adaptive learning rate 
        per parameter.

        The RMSProp class has the following parameters:
        
        :param learning_rate(float): the optimizer learning rate after the forward pass.
        :param current learning rate(float): the previous iteration learning rate.
        :param iterations(int): current iteration.
        :param decay(float): decay rate.
        :param momentum(float): a constant rate that helps the model avoid the local minimum.
        :param epsilon(float): a constant rate offset.
        :param rho(float): cache memory decay rate
        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.283-290]

    """
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        """                         
        The function calculates the current learning rate if the decay is set, in regards to the
        current iteration.

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.298-306]
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Updates the parameters by adding an adaptive learning rate, making the changes smoother
        by adding the moving average of the cache.

        
        Args:
        :param layer(layer type): layer parameter.

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.283-290]
        """
        if not hasattr(layer, 'weight_cache'):
            # If layer does not contain momentum arrays, create them filled with zeros
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.derivated_weights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.derivated_biases**2

        layer.derivated_weights += -self.current_learning_rate * \
            layer.derivated_weights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.derivated_biases += -self.current_learning_rate * \
            layer.derivated_biases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        """
        Increment the number of iterations before each update.

        Args:

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.283-290]
        """
        self.iterations += 1
