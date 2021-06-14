import numpy as np

class Loss:
    """
        The Loss/cost function is a metric which quantifies how wrong the model is,
        its value representing the model error.

        Each object that inherits this class must implement the 'forward' method.

    The Loss class contains:
        :param weights(np.array) : Dense Layer's weights.
        :param biases(np.array) : Dense Layer's biases
        :param weight_regularizer_l1(float) : l1 weight regulizer factor
        :param weight_regularizer_l2(float) : l2 weight regulizer factor
        :param bias_regularizer_l1(float) : l1 bias regulizer factor
        :param bias_regularizer_l2(float) : l2 bias regulizer factor

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.129-139]

    """
    def regularization_loss(self):
        """
        Regularization loss calculation

        Args:

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.129-139]
        """
        regularization_loss = 0
        # Iterate all trainable layers
        for layer in self.trainable_layers:
            # Applying regularizers if they re initialized
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                    np.sum(np.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                    np.sum(layer.weights *
                           layer.weights)
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                    np.sum(np.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                    np.sum(layer.biases *
                           layer.biases)
        return regularization_loss
        
    def remember_trainable_layers(self, trainable_layers):
        """
         Set/remember trainable layers

        Args:

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.129-139]
        """
        self.trainable_layers = trainable_layers
        # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y, include_regularization=False):
        """
        Compute the loss.

        Args:
            output (np.array): model predictions.
            y (np.array): actual values .
            include_regularization(bool): Specify if the return should contain the regularization loss.


        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.129-139]
        """
        # Calculate sample loss
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        # If include_regularization is False.
        if not include_regularization:
            return data_loss
        # Return the data and regularization loss
        return data_loss, self.regularization_loss()

    def calculate_accumulated(self, include_regularization=False):
        """
        Calculates accumulated loss

        Args: include_regularization(bool): Specify if the return should contain the regularization loss.
        
        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.129-139]
        """
        data_loss = self.accumulated_sum / self.accumulated_count
        # If include_regularization is False.
        if not include_regularization:
            return data_loss
        # Return the data and regularization loss
        return data_loss, self.regularization_loss()

    def new_pass(self):
        """
        Reset variables for accumulated loss

        Args:
        
        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.111-129]
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0
