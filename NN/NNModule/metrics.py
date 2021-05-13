import numpy as np
import layers
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
    def __init__(self,  binary=False):
        # Binary mode?
        self.binary = binary
        self.accumulated_count = 0
        self.accumulated_sum =0
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
    dl = layers.Layer_Dense(5, 5, 0.003, 0.003, 0.003, 0.003)
    dl.weights = inputs
    dl.forward(inputs, True)
    dl.backward(inputs)

    l = Accuracy_Categorical(True)
    l.calculate(inputs,inputs)
    print(l.accumulated_sum,l.accumulated_count)
    print(l.calculate_accumulated())
