import numpy as np
from Layers.BackPropagation import BackPropagation
#from tensorflow.keras.layers import BatchNormalization
arr = np.array([1, 6, 5, 7, 4, 3, 2, 5, 6, 3, 2, 4, 5, 3, 2, 6],
               ).reshape((2, 2, 2, 2)).astype(np.float32)


# arr = np.array([1,6,5,7,4,3,2,5,6,3,2,4,5,3,5,6] ).reshape((2,2,2,2))
# b = BackPropagation()

# out,_ = b.forward(arr)

# print(out)
def pure_batch_norm(X, gamma, beta, eps=1e-5):
    if len(X.shape) not in (2, 4):
        raise ValueError('only supports dense or 2dconv')

    # dense
    if len(X.shape) == 2:
        # mini-batch mean
        mean = np.mean(X, axis=0)
        # mini-batch variance
        variance = np.mean((X - mean) ** 2, axis=0)
        # normalize
        X_hat = (X - mean) * 1.0 / np.sqrt(variance + eps)
        # scale anp shift
        out = gamma * X_hat + beta

    # 2d conv
    elif len(X.shape) == 4:
        # extract the dimensions
        N, C, H, W = X.shape
        # mini-batch mean
        mean = np.mean(X, axis=(0, 2, 3))
        # mini-batch variance
        variance = np.mean((X - mean.reshape((1, C, 1, 1)))
                           ** 2, axis=(0, 2, 3))
        # normalize
        X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / \
            np.sqrt(variance.reshape((1, C, 1, 1)) + eps)
        # scale anp shiftg
        out = gamma.reshape((1, C, 1, 1)) * X_hat + beta.reshape((1, C, 1, 1))

    return out


arr1 = np.array([1, 6, 5, 7, 4, 3, 2, 5, 6, 3, 2, 4, 5, 3, 2, 5]
                ).reshape((2, 2, 2, 2)).astype(np.float32)

#arr1 = np.array([1, 7, 5, 4, 6, 10], ).reshape((3, 2)).astype(np.float32)
# arr2 = np.array([1, 7, 5, 4, 6, 10], ).reshape((3, 2)).astype(np.float32)
# b = BatchNormalization()
# out = b(arr1)
# print(out)
# print(pure_batch_norm(arr1,np.array([1,1]).astype(np.float32),np.array([0,0]).astype(np.float32)))

bc = BackPropagation()
out, cache = bc.forward(arr1)
dx, gamma, dbeta = bc.backward(arr1,cache)
print(dx)
print(gamma)
print(dbeta)
