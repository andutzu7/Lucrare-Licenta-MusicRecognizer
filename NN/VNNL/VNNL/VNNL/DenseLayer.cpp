#include "DenseLayer.h"

DenseLayer::DenseLayer(size_t nr_inputs, size_t nr_neurons, float weight_regularizer_l1, float weight_regularizer_l2, float bias_regularizer_l1, float bias_regularizer_l2)
{
    this->weights = 0.01 * xt::random::randn<double>({nr_inputs, nr_neurons});
    this->biases = xt::zeros<double>({1, (int)nr_neurons});
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

// Derivated values is the gradient
void DenseLayer::Backward(xt::xarray<double> derivated_values)
{
    xt::xarray<double> inputs_transposed = xt::transpose(this->inputs);
    this->derivated_weights = xt::linalg::dot(inputs_transposed, derivated_values);

    this->derivated_biases = xt::sum(derivated_values, 0);
    //lambda useful for updating the ones array
    auto regularize = [](xt::xarray<double> weights) {
        auto weights_shape = weights.shape();

        xt::xarray<double> dL = xt::arange(weights.size());
        size_t index = 0;
        for (const auto &value : weights)
        {
            if (value < 0.0)
            {
                dL[index] = -1;
            }
            else
            {
                dL[index] = 1;
            }
            index++;
        }
        dL.reshape(weights.shape());
        return dL;
    };
    regularize(weights);
    //L1 on weights
    if (this->weight_regularizer_l1 > 0)
    {

        xt::xarray<double> dL1 = regularize(this->weights);
        this->derivated_weights += this->weight_regularizer_l1 * dL1;
    }
    // L2 on weights
    if (this->weight_regularizer_l2 > 0)
    {
        this->derivated_weights += 2 * this->weight_regularizer_l2 * this->weights;
    }
    // L1 on biases
    if (this->bias_regularizer_l1 > 0)
    {
        xt::xarray<double> dL1 = regularize(this->biases);
        this->derivated_biases += this->bias_regularizer_l1 * dL1;
        cout<<this->derivated_biases;
    }
    // L2 on biases
    if (this->bias_regularizer_l2 > 0)
    {
        this->derivated_biases += 2 * this->bias_regularizer_l2 * this->biases;
    }
    //   Gradient on values
    xt::xarray<double> weights_transposed = xt::transpose(this->weights);
    this->derivated_inputs = xt::linalg::dot(derivated_values, weights_transposed);
}

std::pair<xt::xarray<double>, xt::xarray<double>> DenseLayer::getParameters()
{
    return std::pair<xt::xarray<double>, xt::xarray<double>>(this->weights, this->biases);
}
void DenseLayer::setParameters(xt::xarray<double> weights, xt::xarray<double> biases)
{
    this->weights = weights;
    this->biases = biases;
}
