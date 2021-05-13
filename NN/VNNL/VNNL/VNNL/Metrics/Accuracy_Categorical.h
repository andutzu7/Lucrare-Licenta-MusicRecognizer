#pragma once
#include "Accuracy.h"

class Accuracy_Categorical : public Accuracy
{
public:
    Accuracy_Categorical(bool binary = false);
    xt::xarray<double> Compare(xt::xarray<double> predictions , xt::xarray<double> actual_values);
//sos
public:
    bool binary;
};