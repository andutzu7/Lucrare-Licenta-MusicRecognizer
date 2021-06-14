import numpy as np
from .Accuracy import Accuracy

# Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):
    """

    Accuracy calculation for regression model by checking if the index
    of the maximal true value is equal to the index of the maximal predicted value.
    
    Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.129-139]
    """
    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        """
        Compares predictions to the ground truth values

        Args:
            predictions (np.array): model predictions.
            y (np.array): actual values .
        
        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.129-139]
        """
        return np.absolute(predictions - y) < self.precision