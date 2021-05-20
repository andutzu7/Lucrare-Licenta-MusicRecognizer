# from kapre.composed import get_melspectrogram_layer
# import pydub 
# import numpy as np
# from scipy.io.wavfile import read

# sr,data = read("anotherbrick.wav")
# data = data.reshape(-1,1)
            

# input_shape = (int(16000*1.0), 1)
# i = get_melspectrogram_layer(input_shape=input_shape,
#                              pad_end=True,
#                              sample_rate=16000,
#                              return_decibel=True,
#                              input_data_format='channels_last',
#                              output_data_format='channels_last')
                            

# print(i(data))

from Layers.Conv2DLayer import Conv2D
from Layers.MaxPooling2DLayer import MaxPooling2D
from Layers.DropoutLayer import DropoutLayer
import cv2

c2d = Conv2D(3,(2,2))

image = cv2.imread('pants.png')
image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))
out = c2d.forward_pass(image)
print(out)
mp2d = MaxPooling2D(pool_shape=(2,2))
out2 = mp2d.forward_pass(out)

dl = DropoutLayer(0.2)
dl.forward(out2,True)
print(dl.output)
