import os
import pickle
import numpy as np
import cv2

from tensorflow.keras.layers import Conv2D
from Layers.DenseLayer import DenseLayer
from Activations.ActivationSoftmax import ActivationSoftmax
from Activations.ActivationReLu import ActivationReLu
from Optimizers.OptimizerAdam import OptimizerAdam
from Metrics.CategoricalCrossentropy import CategoricalCrossentropy
from Metrics.CategoricalAccuracy import CategoricalAccuracy
from ModelClass.Model import Model

def load_mnist_dataset(dataset, path):
    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))
    # Create lists for samples and labels
    X = []
    y = []
    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(
                os.path.join(path, dataset, label, file),
                cv2.IMREAD_UNCHANGED)
            # And append it and a label to the lists
            X.append(image)
            y.append(label)
    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')
# MNIST dataset (train + test)


def create_data_mnist(path):
    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    # And return all the data
    return X, y, X_test, y_test


# Label index to label name relation
fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

if __name__ == "__main__":
    # X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
    # # Shuffle the training dataset
    # keys = np.array(range(X.shape[0]))
    # np.random.shuffle(keys)
    # X = X[keys]
    # y = y[keys]
    # X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    # X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
    #           127.5) / 127.5
    # # Instantiate the model
    # model = Model()
    # # Add layers
    # model.add(DenseLayer(X.shape[1], 64))
    # model.add(ActivationReLu())
    # model.add(DenseLayer(64, 64))
    # model.add(ActivationReLu())
    # model.add(DenseLayer(64, 10))
    # model.add(ActivationSoftmax())
    # # Set loss, optimizer and accuracy objects
    # model.set(
    #     loss=CategoricalCrossentropy(),
    #     optimizer=OptimizerAdam(decay=5e-5),
    #     accuracy=CategoricalAccuracy()
    # )
    # # Finalize the model
    # model.finalize()
    # # Train the model
    # model.train(X, y, validation_data=(X_test, y_test),
    #             epochs=10, batch_size=128, print_every=100)
    # model.save('fashion_mnist.model')
    # # Read an image
    image_data = cv2.imread('pants.png', cv2.IMREAD_GRAYSCALE)
    # Resize to the same size as Fashion MNIST images
#    image_data = cv2.resize(image_data, (28, 28))
    # Invert image colors
    image_data = 255 - image_data
    # Reshape and scale pixel data
    image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
    # Load the model
    model = Model.load('fashion_mnist.model')
    # Predict on the image
    confidences = model.predict(image_data)
    # Get prediction instead of confidence levels
    predictions = model.output_layer_activation.predictions(confidences)
    # Get label name from label index
    prediction = fashion_mnist_labels[predictions[0]]
    print(prediction)
