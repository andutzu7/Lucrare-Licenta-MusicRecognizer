import numpy as np

class Accuracy:
    """
        The Accuracy is a metric which describes how often the largest condifence
        is the correct class.
        This is the abstract/generic class which will be inherited by the categorical
        and regression accuracy.
        Each object that inherits this class must implement the 'compare' method.

    The Accuracy class contains:
        :param accumulated_sum(float) : accumulated sum 
        :param accumulated_count(float) : accumulated count

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.129-139]

    """
    def calculate(self, predictions, y):
        """
        Calculates an accuracy given predictions and ground truth values

        Args:
            predictions (np.array): model predictions.
            y (np.array): actual values .
        
        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.129-139]
        """
        # Compare predictions to the truth values.
        comparisons = self.compare(predictions, y)
        # Calculate the accuracy
        accuracy = np.mean(comparisons)
        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy

    def calculate_accumulated(self):
        """
        Compute the accumlated accuracy

        Args:
        
        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.129-139]
        """
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy

    def new_pass(self):
        """
        Reset variables for accumulated accuracy

        Args:
        
        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.129-139]
        """
        self.accumulated_sum = 0
        self.accumulated_count = 0