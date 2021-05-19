import numpy as np
import cv2
from tensorflow.keras.layers import Conv2D as conv2d
from Layers.Conv2D import Conv2D 
#from tensorflow.keras.layers import Conv2D


# arr = cv2.imread("./pants.png")
# arr = arr.reshape((1,28,28,3)).astype(np.float32)
# c2dd = conv2d(8,(3,3),padding='same',input_shape = (28,28,3))
# print(c2dd(arr))

# c2d = Conv2D(8,(3,3),(3,28,28))
# arr = cv2.imread("./pants.png")
# arr = np.moveaxis(arr, -1, 0)

# arr = arr.reshape((1,3,28,28)).astype(np.float32)
# print(c2d.forward_pass(arr))
