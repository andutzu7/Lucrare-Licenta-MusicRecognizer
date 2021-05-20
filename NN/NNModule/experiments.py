import numpy as np
import cv2
from Layers.MaxPooling2DLayer import MaxPooling2D
#from tensorflow.keras.layers import MaxPooling2D as mp2d
# arr = cv2.imread("./pants.png")
# arr = arr.reshape((1,28,28,3)).astype(np.float32)
# c2dd = mp2d((3,3),strides=1,padding='same',input_shape = (28,28,3))
# print(c2dd(arr).shape)

arr = cv2.imread("./pants.png")
arr = np.moveaxis(arr, -1, 0)


arr = arr.reshape((1,3,28,28)).astype(np.float32)
c2d = MaxPooling2D((3,3),stride = 2,input_shape = (3,28,28))
print(c2d.forward_pass(arr).shape)
