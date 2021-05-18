import numpy as np


class BackPropagation:

    def __init__(self):
        self.gamma = None
        self.beta = None
        self.moving_mean = None
        self.moving_variation = None
        pass

    def forward(self, x, epsilon=0.001, momentum=0.999, training=True):
        cache = None
        if len(x.shape) == 2:
            _, D = x.shape

            if self.moving_mean is None and self.moving_variation is None:
                self.moving_mean = np.zeros(D, dtype=x.dtype)
                self.moving_variation = np.ones(D, dtype=x.dtype)

            if self.gamma is None and self.beta is None:
                self.gamma = np.ones(D, dtype=x.dtype)
                self.beta = np.zeros(D, dtype=x.dtype)

        elif len(x.shape) == 4:

            N, C, H, W = x.shape

            if self.moving_mean is None and self.moving_variation is None:
                self.moving_mean = np.zeros((1, C, 1, 1), dtype=x.dtype)
                self.moving_variation = np.ones((1, C, 1, 1), dtype=x.dtype)

            if self.gamma is None and self.beta is None:
                self.gamma = np.ones((1, C, 1, 1), dtype=x.dtype)
                self.beta = np.zeros((1, C, 1, 1), dtype=x.dtype)

        moving_mean = self.moving_mean
        moving_variation = self.moving_variation

        gamma = self.gamma
        beta = self.beta

  # `gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta`, where:
        if training == True:
            # Dense layer
            if len(x.shape) == 2:
                sample_mean = x.mean(axis=0)
                sample_var = x.var(axis=0)

                moving_mean = momentum * moving_mean + \
                    (1 - momentum) * sample_mean
                moving_variation = momentum * \
                    moving_variation + (1 - momentum) * sample_var

                standard_deviation = np.sqrt(sample_var + epsilon)
                x_centered = x - sample_mean
                x_norm = x_centered / standard_deviation
                out = gamma * x_norm + beta
            # 2D Conv
            elif len(x.shape) == 4:

                sample_mean = x.mean(axis=(0, 2, 3))
                sample_var = x.var(axis=(0, 2, 3))

                moving_mean = momentum * moving_mean + \
                    (1 - momentum) * sample_mean.reshape((1, C, 1, 1))
                moving_variation = momentum * \
                    moving_variation + (1 - momentum) * \
                    sample_var.reshape((1, C, 1, 1))

                standard_deviation = np.sqrt(
                    sample_var.reshape((1, C, 1, 1)) + epsilon)
                x_centered = x - sample_mean.reshape((1, C, 1, 1))
                x_norm = x_centered / standard_deviation
                out = gamma * x_norm + beta

            cache = (x_norm, x_centered, standard_deviation, gamma)

  # `gamma * (batch - self.moving_mean) / sqrt(self.moving_var + epsilon) + beta`.
        elif training == False:
            x_norm = (x - moving_mean) / np.sqrt(moving_variation + epsilon)
            out = gamma * x_norm + beta

        self.moving_mean = moving_mean
        self.moving_variation = moving_variation

        return out, cache

    def backward(self, derivated_outputs, cache):
        if len(derivated_outputs.shape) == 2:
            N = derivated_outputs.shape[0]
            x_norm, x_centered, std, gamma = cache

            derivated_gamma = (derivated_outputs * x_norm).sum(axis=0)
            derivated_beta = derivated_outputs.sum(axis=0)

            dx_norm = derivated_outputs * gamma
            dx = 1/N / std * (N * dx_norm -
                              dx_norm.sum(axis=0) -
                              x_norm * (dx_norm * x_norm).sum(axis=0))
        elif len(derivated_outputs.shape) == 4:

            N, C, H, W = derivated_outputs.shape
            x_norm, x_centered, std, gamma = cache

            derivated_gamma = (derivated_outputs * x_norm).sum(axis=(0,2,3))
            derivated_beta = derivated_outputs.sum(axis=(0,2,3))

            dx_norm = derivated_outputs * gamma
            dx = 1/C / std * (C * dx_norm -
                              dx_norm.sum(axis=(0,2,3)) -
                              x_norm * (dx_norm * x_norm).sum(axis=(0,2,3)))

        return dx, derivated_gamma, derivated_beta
