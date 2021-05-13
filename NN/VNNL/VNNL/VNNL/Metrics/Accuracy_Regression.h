#pragma once
#include "Accuracy.h"

class Accuracy_Regression : public Accuracy
{
public:
    Accuracy_Regression(xt::xarray<double> actual_values,bool reinit = true);
    xt::xarray<double> Compare(xt::xarray<double> predictions , xt::xarray<double> actual_values);
//sos
public:
    double precision = 0.0;
};