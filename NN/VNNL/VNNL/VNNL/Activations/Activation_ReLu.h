#pragma once
#include "Activation.h"

class Activation_ReLu : public Activation
{

public:
    void Forward(xt::xarray<double> inputs);
    void Backward(xt::xarray<double> derivated_values);
    xt::xarray<double> Prediction(xt::xarray<double>outputs);
 //one day make these private
};