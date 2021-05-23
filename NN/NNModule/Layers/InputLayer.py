from kapre.composed import get_melspectrogram_layer
import numpy as np
# Input layer specifically for the musical instruments classification problem


class InputLayer:
    # Forward pass
    def forward(self, inputs):
        input_shape = (int(16000*1.0), 1)
        i = get_melspectrogram_layer(input_shape=input_shape,
                                     pad_end=True,
                                     sample_rate=16000,
                                     return_decibel=True,
                                     input_data_format='channels_last',
                                     output_data_format='channels_first')
        self.output = np.array(i(inputs)).astype(np.float32)

