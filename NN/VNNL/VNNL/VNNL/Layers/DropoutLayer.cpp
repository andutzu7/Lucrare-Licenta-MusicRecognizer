#include "DropoutLayer.h"
DropoutLayer::DropoutLayer(double rate)
{
    this->rate = 1 - rate;
}

void DropoutLayer::Forward(xt::xarray<double> inputs, bool training)
{
    this->inputs = inputs;
    if (!training)
    {
        this->outputs = xt::xarray<double>(inputs);
    }
    else
    {
        this->binary_mask = (xt::random::binomial(inputs.shape(), 1, this->rate)) / this->rate;
        this->outputs = inputs * binary_mask;
    }
}
void DropoutLayer::Backward(xt::xarray<double> derivated_values)
{
    this->derivated_inputs = derivated_values * this->binary_mask;
}