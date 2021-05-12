#include "Activation_ReLu.h"

void Activation_ReLu::Forward(xt::xarray<double> inputs)
{
    this->inputs = inputs;
    this->outputs = xt::maximum(0,inputs);
}

void Activation_ReLu::Backward(xt::xarray<double> derivated_values)
{
    this->derivated_inputs = derivated_values;

    for(size_t i = 0 ; i < inputs.size() ; i++)
    {
        if (this->inputs[i] <=0)
        {
             this->derivated_inputs[i] = 0;
        }
    }
}
xt::xarray<double> Activation_ReLu::Prediction(xt::xarray<double> outputs)
{
    return outputs;
}