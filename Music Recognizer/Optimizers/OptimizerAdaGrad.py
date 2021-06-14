import numpy as np

class OptimizerAdagrad:
    """
        Adagrad(adaptive gradient) performs a per-parameter learning rate in order to normalize the updates
        made to the features.

        The Adagrad has the following parameters:
        
        :param learning_rate(float): the optimizer learning rate after the forward pass.
        :param current learning rate(float): the previous iteration learning rate.
        :param decay(float): decay rate.
        :param epsilon(float): a constant rate offset.
        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.291-296]

    """

    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        """                         
        The function calculates the current learning rate if the decay is set, in regards to the
        current iteration.

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.291-296]
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Updates the parameters.

        
        Args:
        :param layer(layer type): layer parameter.

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.291-296]
        """
        # If layer does not contain cache arrays, create them filled with zeros.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        # Update cache with squared current gradients
        layer.weight_cache += layer.derivated_weights**2
        layer.bias_cache += layer.derivated_biases**2
        # SGD parameter update and normalization
        layer.weights += -self.current_learning_rate * \
            layer.derivated_weights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.derivated_biases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)


    def post_update_params(self):
        """
        Increment the number of iterations before each update.

        Args:

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.291-296]
        """
        self.iterations += 1
