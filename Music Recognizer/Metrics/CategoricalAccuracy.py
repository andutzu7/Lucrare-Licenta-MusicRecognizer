import numpy as np
from .Accuracy import Accuracy

class CategoricalAccuracy(Accuracy):
    """

    Accuracy calculation for classification model by checking if the index
    of the maximal true value is equal to the index of the maximal predicted value.
    
    Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.129-139]
    """
    def __init__(self, binary=False):
        self.binary = binary
    def init(self, y):
        pass
    def compare(self, predictions, y):
        """
        Compares predictions to the ground truth values

        Args:
            predictions (np.array): model predictions.
            y (np.array): actual values .
        
        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.129-139]
        """
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y

