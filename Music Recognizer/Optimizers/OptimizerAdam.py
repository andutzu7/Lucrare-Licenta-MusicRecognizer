from Layers.Conv2DLayer import Conv2D
import numpy as np

class OptimizerAdam:
    """
    Adam(Adaptive Momentum) is built on top of RMSProp applying a momentum concept like SGD.
    Instead of applying the current gradients to the parameters, it adds the momentum like in th SGD optimizer,
    applying a per-weight adaptive learning.

    :param learning_rate(float): the optimizer learning rate after the forward pass.
    :param current learning rate(float): the previous iteration learning rate.
    :param decay(float): decay rate.
    :param iterations(int): current iteration.
    :param epsilon(float): a constant rate offset.
    Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.304-309]
    """
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        """                         
        The function calculates the current learning rate if the decay is set, in regards to the
        current iteration.

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.304-309]
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Updates the parameters.

        
        Args:
        :param layer(layer type): layer parameter.

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.304-309]
        """
        # If layer does not contain momentum arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * \
            layer.weight_momentums + \
            (1 - self.beta_1) * layer.derivated_weights
        layer.bias_momentums = self.beta_1 * \
            layer.bias_momentums + \
            (1 - self.beta_1) * layer.derivated_biases
        # Get corrected momentum
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.derivated_weights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.derivated_biases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        
        # Update weights and biases
        layer.weights += -self.current_learning_rate * \
            weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) +
             self.epsilon)
        layer.biases += -self.current_learning_rate * \
            bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) +
             self.epsilon)

    def post_update_params(self):
        """                         
        The function calculates the current learning rate if the decay is set, in regards to the
        current iteration.

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.304-309]
        """
        self.iterations += 1
