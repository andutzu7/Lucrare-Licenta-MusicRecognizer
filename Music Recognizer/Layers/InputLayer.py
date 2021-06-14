from kapre.composed import get_melspectrogram_layer
import numpy as np

# Input layer specifically for the musical instruments classification problem
class InputLayer:
    """
        The InputLayer class takes as an input a batch of data,transforms it using the get_melspectogram layer
        and returns the output in form of a numpy array.

    The DenseLayer class contains:
        :param output(np.array) : The outputs of the layer after the forward pass
    Sources:    *https://www.youtube.com/user/seth8141/featured
                *https://github.com/seth814/Audio-Classification
                *https://kapre.readthedocs.io/en/latest/composed.html
    """
    # Forward pass
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        """
        Performs the forward pass by feeding the inputs to the composed kapre layer which consists of a
        keras.Sequential model consists of STFT, Magnitude, ApplyFilterbank(_mel_filterbank) operation
        thus resulting in a 2D Numpy array which contains the MelSpectogram of the audio input file.
        Args:
            inputs (np.array): given inputs.

        Sources:    * Neural Networks from Scratch - Harrison Kinsley & Daniel Kukieła [pg.66-71]
        """

        # Predefined sample output shape
        input_shape = (int(16000*1.0), 1)
        # Applying the transformation
        i = get_melspectrogram_layer(input_shape=input_shape,
                                     pad_end=True,
                                     sample_rate=16000,
                                     return_decibel=True,
                                     input_data_format='channels_last',
                                     output_data_format='channels_first')
        # Returning the output
        self.output = np.array(i(inputs)).astype(np.float32)


