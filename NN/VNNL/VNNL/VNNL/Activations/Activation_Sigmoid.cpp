#include "Activation_Sigmoid.h"

void Activation_Sigmoid::Forward(xt::xarray<double> inputs)
{
    this->inputs = inputs;
    this->outputs = 1 / (1 + xt::exp(-inputs));
}

void Activation_Sigmoid::Backward(xt::xarray<double> derivated_values)
{
    this->derivated_inputs = derivated_values * (1 - this->outputs) * this->outputs;
}
xt::xarray<double> Activation_Sigmoid::Prediction(xt::xarray<double> outputs)
{
    for (size_t index = 0 ; index <outputs.size();index++)
    {
        if(outputs[index] >0.5)
        {
            outputs[index] = 1;
        }
        else
        {
            outputs[index] = 0;
        }
    }
    return outputs;
}