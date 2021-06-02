# Import modules
from Layers.Conv2DLayer import Conv2D
import copy
import cv2
import os
import numpy as np
import copy
import pickle
# Import clasess
from Layers.InputLayer import InputLayer
from .DataGenerator import DataGenerator
from Metrics.CategoricalCrossentropy import CategoricalCrossentropy
from Activations.ActivationSoftmax import ActivationSoftmax
from Metrics.ActivationSoftmaxCategoricalCrossentropy import Activation_Softmax_Loss_CategoricalCrossentropy
# Model class


class Model:
    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None
    # Add objects to the model

    def add(self, layer):
        self.layers.append(layer)
    # Set loss, optimizer and accuracy

    def set(self, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy
    # Finalize the model

    def finalize(self):
        # Create and set the input layer
        self.input_layer = InputLayer()
        # Count all the objects
        layer_count = len(self.layers)
        # Initialize a list containing trainable layers:
        self.trainable_layers = []
        # Iterate the objects
        for i in range(layer_count):
            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            # The last layer - the next object is the loss
            # Also let's save aside the reference to the last object
            # whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        # Update loss object with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(
                self.trainable_layers
            )
        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], ActivationSoftmax) and \
           isinstance(self.loss, CategoricalCrossentropy):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = \
                Activation_Softmax_Loss_CategoricalCrossentropy()
    # Train the model

    def train(self, X=None, y=None, train_generator=None, validation_generator=None, epochs=1, batch_size=None,
              print_every=1, validation_data=None):
        # Initialize accuracy object
        if y is not None:
            self.accuracy.init(y)
        # Default value if batch size is not being set
        train_steps = 1
        # Calculate number of steps
        if X is not None:
            if batch_size is not None:
                train_steps = len(X) // batch_size
                # Dividing rounds down. If there are some remaining
                # data but not a full batch, this won't include it
                # Add `1` to include this not full batch
                if train_steps * batch_size < len(X):
                    train_steps += 1
        elif train_generator is not None:
            train_steps = 50#int(train_generator.__len__()/batch_size)
        # Main training loop
        for epoch in range(1, epochs+1):
            # Print epoch number
            print(f'epoch: {epoch}')
            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()
            # Iterate over steps
            for step in range(train_steps):
                # If batch size is not set -
                # train using one step and full dataset
                if X is not None:
                    if batch_size is None:
                        batch_X = X
                        batch_y = y
                # Otherwise slice a batch
                    else:
                        batch_X = X[step*batch_size:(step+1)*batch_size]
                        batch_y = y[step*batch_size:(step+1)*batch_size]
                elif train_generator is not None:
                    batch_X, batch_y = train_generator.__getitem__(step)
                    self.accuracy.init(batch_y)
                # Perform the forward pass
                output = self.forward(batch_X)
                # Calculate loss
                # de pus aici regularisation loss
                data_loss = \
                    self.loss.calculate(output, batch_y,
                                        include_regularization=False)
                loss = data_loss
                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(
                    output)
                accuracy = self.accuracy.calculate(predictions,
                                                   batch_y)
                # Perform backward pass
                self.backward(output, batch_y)
                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                # Print a summary
                if not step % print_every or step == train_steps - 1:
                    message = f'step: {step},  acc: {accuracy:.3f},  loss: {loss:.3f} ( + data_loss: {data_loss:.3f}),lr: {self.optimizer.current_learning_rate} \n'
                    with open('logs','a+') as f:
                        f.write(message)
                    print(message)
            # Get and print epoch loss and accuracy
            epoch_data_loss = \
                self.loss.calculate_accumulated(
                    include_regularization=False)
            epoch_loss = epoch_data_loss 
            epoch_accuracy = self.accuracy.calculate_accumulated()
            message = f'training,  acc: {epoch_accuracy:.3f},loss: {epoch_loss:.3f} data_loss: {epoch_data_loss:.3f},lr: {self.optimizer.current_learning_rate})\n'
            print(message)
            with open('./logs2','a+') as f:
                f.write(message)
            # If there is the validation data
            if validation_data is not None:
                # Evaluate the model:
                self.evaluate(*validation_data,
                              batch_size=batch_size)
            elif validation_generator is not None:
                self.evaluate(validation_generator=validation_generator,
                              batch_size=batch_size)
    # Evaluates the model using passed-in dataset

    def evaluate(self, X_val=None, y_val=None, validation_generator=None, batch_size=None):
        # Default value if batch size is not being set
        validation_steps = 1
        # Calculate number of steps
        if X_val is not None:
            if batch_size is not None:
                validation_steps = len(X_val) // batch_size
                # Dividing rounds down. If there are some remaining
                # data but not a full batch, this won't include it
                # Add `1` to include this not full batch
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1
        else:
            if validation_generator is not None:
                validation_steps = 50#int(validation_generator.__len__()/batch_size)
        # Reset accumulated values in loss
        # and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()
        # Iterate over steps
        for step in range(validation_steps):
            # If batch size is not set -
            # train using one step and full dataset
            if X_val is not None:
                if batch_size is None:
                    batch_X = X_val
                    batch_y = y_val
                # Otherwise slice a batch
                else:
                    batch_X = X_val[
                        step*batch_size:(step+1)*batch_size
                    ]
                    batch_y = y_val[
                        step*batch_size:(step+1)*batch_size
                    ]
            elif validation_generator is not None:
                batch_X, batch_y = validation_generator.__getitem__(step)
                # Perform the forward pass
            output = self.forward(batch_X)
            # Calculate the loss
            self.loss.calculate(output, batch_y)
            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(
                output)
            self.accuracy.calculate(predictions, batch_y)
        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        # Print a summary
        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')
    # Predicts on the samples

    def predict(self, X=None, test_generator=None,  batch_size=None):
        # Default value if batch size is not being set
        prediction_steps = 1
        # Calculate number of steps
        if X is not None:
            if batch_size is not None:
                prediction_steps = len(X) // batch_size
                # Dividing rounds down. If there are some remaining
                # data but not a full batch, this won't include it
                # Add `1` to include this not full batch
                if prediction_steps * batch_size < len(X):
                    prediction_steps += 1
        elif test_generator is not None:
            prediction_steps = test_generator.__len__()
        # Model outputs
        output = []
        # Iterate over steps
        for step in range(prediction_steps):
            # If batch size is not set -
            # train using one step and full dataset
            if X is not None:
                if batch_size is None:
                    batch_X = X
                # Otherwise slice a batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
            elif test_generator is not None:
                batch_X = test_generator.__getitem__(step)
                # Perform the forward pass
            batch_output = self.forward(batch_X)
            # Append batch prediction to the list of predictions
            output.append(batch_output)
        # Stack and return results
        return np.vstack(output)
    # Performs forward pass

    def forward(self, X):
        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X)
        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output)
        # "layer" is now the last object from the list,
        # return its output
        return layer.output
    # Performs backward pass

    def backward(self, output, y):
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set derivated_inputs property
            self.softmax_classifier_output.backward(output, y)
            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set derivated_inputs in this object
            self.layers[-1].derivated_inputs = \
                self.softmax_classifier_output.derivated_inputs
            # Call backward method going through
            # all the objects but last
            # in reversed order passing derivated_inputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.derivated_inputs)
            return
        # First call backward method on the loss
        # this will set derivated_inputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)
        # Call backward method going through all the objects
        # in reversed order passing derivated_inputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.derivated_inputs)
    # Retrieves and returns parameters of trainable layers

    def get_parameters(self):
        # Create a list for parameters
        parameters = []
        # Iterable trainable layers and get their parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        # Return a list
        return parameters
    # Updates the model with new parameters

    def set_parameters(self, parameters):
        # Iterate over the parameters and layers
        # and update each layers with each set of the parameters
        for parameter_set, layer in zip(parameters,
                                        self.trainable_layers):
            layer.set_parameters(*parameter_set)
    # Saves the parameters to a file

    def save_parameters(self, path):
        # Open a file in the binary-write mode
        # and save parameters into it
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
    # Loads the weights and updates a model instance with them

    def load_parameters(self, path):
        # Open file in the binary-read mode,
        # load weights and update trainable layers
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))
    # Saves the model

    def save(self, path):
        # Make a deep copy of current model instance
        model = copy.deepcopy(self)
        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()
        # Remove data from the input layer
        # and gradients from the loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('derivated_inputs', None)
        # For each layer remove inputs, output and derivated_inputs properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'derivated_inputs',
                             'derivated_weights', 'derivated_biases']:
                layer.__dict__.pop(property, None)
        # Open a file in the binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    # Loads and returns a model

    @staticmethod
    def load(path):
        # Open file in the binary-read mode, load a model
        with open(path, 'rb') as f:
            model = pickle.load(f)
        # Return a model
        return model
