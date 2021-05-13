#pragma once
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xsort.hpp>
#include <math.h>


class Accuracy
{
public:
    virtual xt::xarray<double> Compare(xt::xarray<double> predictions , xt::xarray<double> actual_values) = 0;
    double Calculate(xt::xarray<double> predictions , xt::xarray<double> actual_values);
    double Calculate_accumulated();
    void New_Pass();
//sos
public:
    double accumulation_sum =0.0;
    int accumulation_count =0;
};