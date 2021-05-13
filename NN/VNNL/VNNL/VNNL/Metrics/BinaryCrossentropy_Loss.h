#pragma once
#include "Loss.h"

class BinaryCrossentropy : public Loss
{

public:
    xt::xarray<double> Forward(xt::xarray<double> predictions, xt::xarray<double> actual_values);
    void Backward(xt::xarray<double> derivated_values, xt::xarray<double> actual_values);
public:
};