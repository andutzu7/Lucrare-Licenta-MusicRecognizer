#pragma once
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>

class DropoutLayer
{
public:
	DropoutLayer(double rate);
	void Forward(xt::xarray<double> inputs,bool training = true);
	void Backward(xt::xarray<double> derivated_values);

//sa nu uit sa fac zona asta privata
public:
    double rate;
	xt::xarray<double> inputs;
	xt::xarray<double> outputs;
	xt::xarray<double> binary_mask;
	xt::xarray<double> derivated_inputs; 
	
};