
#include "GuerreFroide.h"

DenseLayer::DenseLayer(size_t nr_inputs, size_t nr_neurons, float weight_regularizer_l1, float weight_regularizer_l2, float bias_regularizer_l1, float bias_regularizer_l2)
{
	this->weights = 0.01 * xt::random::randn<double>({ nr_inputs,nr_neurons });
	this->biases = xt::zeros<double>({0, (int) nr_neurons });
    this->weight_regularizer_l1 = weight_regularizer_l1;
    this->weight_regularizer_l2 = weight_regularizer_l2; 
    this->bias_regularizer_l1 = bias_regularizer_l1;
    this->bias_regularizer_l2 = bias_regularizer_l2;

}

void DenseLayer::Forward(xt::xarray<double> inputs)
{
    this->inputs = inputs;

    this->outputs = xt::linalg::dot(inputs, this->weights) + this->biases;
}

void DenseLayer::Backward(xt::xarray<double> gradient_values)
{/*
        auto inputs_transposed = xt::transpose(this->inputs);
        this->derivated_weights = xt::linalg::dot(inputs_transposed, gradient_values);
//de verificat daca se pastreaza dimensiunile
        this->derivated_values = xt::sum(gradient_values);
//compiler workaround
        xt::sum();
        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
    dL1 = np.ones_like(self.weights)
        dL1[self.weights < 0] = -1
        self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
    self.dweights += 2 * self.weight_regularizer_l2 * \
        self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
    dL1 = np.ones_like(self.biases)
        dL1[self.biases < 0] = -1
        self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
    self.dbiases += 2 * self.bias_regularizer_l2 * \
        self.biases
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        # Retrieve layer parameters
        def get_parameters(self) :
        return self.weights, self.biases
        # Set weightsand biases in a layer instance
        def set_parameters(self, weights, biases) :
        self.weights = weights
        self.biases = biases

*/
}
