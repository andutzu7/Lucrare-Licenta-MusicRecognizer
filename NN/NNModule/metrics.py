from layers import Layer_Dense
import numpy as np
# Common loss class


class Loss:
    # Regularization loss calculation
    def regularization_loss(self):
        # 0 by default
        regularization_loss = 0
        # Calculate regularization loss
        # iterate all trainable layers
        for layer in self.trainable_layers:
            # L1 regularization - weights
            # calculate only when factor greater than 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                    np.sum(np.abs(layer.weights))
            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                    np.sum(layer.weights *
                           layer.weights)
            # L1 regularization - biases
            # calculate only when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                    np.sum(np.abs(layer.biases))
            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                    np.sum(layer.biases *
                           layer.biases)
        return regularization_loss
    # Set/remember trainable layers

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
    # Calculates the data and regularization losses
    # given model output and ground truth values

    def calculate(self, output, y, *, include_regularization=False):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        # If just data loss - return it
        if not include_regularization:
            return data_loss
        # Return the data and regularization losses
        return data_loss, self.regularization_loss()
    # Calculates accumulated loss

    def calculate_accumulated(self, *, include_regularization=False):
        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count
        # If just data loss - return it
        if not include_regularization:
            return data_loss
        # Return the data and regularization losses
        return data_loss, self.regularization_loss()
    # Reset variables for accumulated loss

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


class Loss_MeanSquaredError(Loss):  # L2 loss
    # Forward pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        # Return losses
        return sample_losses
    # Backward pass

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
# Mean Absolute Error loss


class Loss_MeanAbsoluteError(Loss):  # L1 loss
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        # Return losses
        return sample_losses
    # Backward pass

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])
        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
# Common accuracy class


class Accuracy:
    # Calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):
        # Get comparison results
        comparisons = self.compare(predictions, y)
        # Calculate an accuracy
        accuracy = np.mean(comparisons)
        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        # Return accuracy
        return accuracy
    # Calculates accumulated accuracy

    def calculate_accumulated(self):
        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count
        # Return the data and regularization losses
        return accuracy
    # Reset variables for accumulated accuracy

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
# Accuracy calculation for classification model


class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        # Binary mode?
        self.binary = binary
    # No initialization is needed

    def init(self, y):
        pass
    # Compares predictions to the ground truth values

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
# Accuracy calculation for regression model


class Accuracy_Regression(Accuracy):
    def __init__(self):
        # Create precision property
        self.precision = None
    # Calculates precision value
    # based on passed-in ground truth values

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    # Compares predictions to the ground truth values

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


if __name__ == "__main__":

    inputs = np.array([[0.01764052,  0.00400157,  0.00978738,  0.02240893,  0.01867558],
                       [-0.00977278, 0.00950088, -
                           0.00151357, -0.00103219,  0.00410599],
                       [0.00144044,  0.01454273,  0.00761038,
                           0.00121675,  0.00443863],
                       [0.00333674,  0.01494079, -0.00205158,
                           0.00313068, -0.00854096],
                       [-0.0255299,  0.00653619,  0.00864436, -0.00742165,  0.02269755]])
    dl = Layer_Dense(5, 5, 0.003, 0.003, 0.003, 0.003)
    dl.weights = inputs
    dl.forward(inputs, True)
    dl.backward(inputs)

    l = Loss_MeanSquaredError()

    print(l.forward(inputs,inputs))
    l.backward(inputs,inputs)
    print(l.dinputs)
   #l.backward(np.array([1,2,3,4],[1,2,3,4]), np.array([0,1,2,4][1,2,3,4],))
