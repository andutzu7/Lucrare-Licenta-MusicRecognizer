#pragma once
#include "Activation.h"

class Activation_Softmax : public Activation
{

public:
    void Forward(xt::xarray<double> inputs);
    void Backward(xt::xarray<double> derivated_values);
    xt::xarray<double> Prediction(xt::xarray<double>outputs);
 //one day make these private
 public:
  xt::xarray<double> gunoi1;
  xt::xarray<double> gunoi2;
};