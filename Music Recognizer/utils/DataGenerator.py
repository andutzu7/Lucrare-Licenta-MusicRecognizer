import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from scipy.io import wavfile
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DataGenerator(tf.keras.utils.Sequence):
    """
    DataGenerator is an object that yields batches of data. It imitates the 
    behaviour of its parent class,tf.keras.utils.Sequence(also a data generator).

    Methods: *__len__(): Return the number of data items
             *__getitem__(index): Return a batch of data
             *on_epoch_end(): Shuffle the data and indexes of the generator

    Args:    *wav_path(list): List of file paths
             *labels(list): List of the file labels
             *sr(int): Audio files sample rate
             *dt(int): File duration time
             *n_classes(int): number of data classes
             *batch_size(int):mini batch size
             *shuffle(Bool):boolean which specifies whether or not to shuffle the data.

    Sources: *https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
             *https://github.com/seth814/Audio-Classification
    
    """
    def __init__(self, wav_paths, labels, sr, dt, n_classes,
                 batch_size=32, shuffle=True):
        self.wav_paths = wav_paths
        self.labels = labels
        self.sr = sr
        self.dt = dt
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = True
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the total number of items in the generator.
        Args:

        Sources: *https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
                 *https://github.com/seth814/Audio-Classification
        """
        return int(np.floor(len(self.wav_paths))) 

    def __getitem__(self, index):
        """
        Returns a batch of data.
        Args: index(int): The index of the data batch

        Sources: *https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
                 *https://github.com/seth814/Audio-Classification
        """
        # Computing the indexes of the current batch(specified with the index parameter)
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        # Read the wav paths and labels
        wav_paths = [self.wav_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # Generate a batch of time data
        X = np.empty((self.batch_size, int(self.sr * self.dt), 1),
                     dtype=np.float32)
        Y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

        # For file, reshape it and transform it to categorical(binary matrix) form
        for i, (path, label) in enumerate(zip(wav_paths, labels)):
            rate, wav = wavfile.read(path)
            X[i,] = wav.reshape(-1, 1)
            Y[i,] = to_categorical(label, num_classes=self.n_classes)

        # Return the data batch
        return X, Y

    def on_epoch_end(self):
        """
        Shuffles the data after the epoch end

        Sources: *https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
                 *https://github.com/seth814/Audio-Classification
        """
        self.indexes = np.arange(len(self.wav_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
