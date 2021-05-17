import numpy as np
from .Accuracy import Accuracy

# Accuracy calculation for classification model
class CategoricalAccuracy(Accuracy):
    def __init__(self, binary=False):
        # Binary mode?
        self.binary = binary
    def init(self, y):
        # Binary mode?
        pass
    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

