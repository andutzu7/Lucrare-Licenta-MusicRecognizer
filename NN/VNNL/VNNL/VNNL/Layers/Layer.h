#pragma once
#include <xtensor/xarray.hpp>
#include <utility>

class Layer
{
public:

	void Forward(xt::xarray<double> inputs);
	void Backward(xt::xarray<double> derivated_values,bool training = true);
	std::pair<xt::xarray<double>,xt::xarray<double>> getParameters();

//but to make it private
public:
	xt::xarray<double> inputs;
	xt::xarray<double> weights;
	xt::xarray<double> biases;
	xt::xarray<double> outputs;

// also only valable in some cases
	xt::xarray<double> derivated_biases;
	xt::xarray<double> derivated_weights;
	xt::xarray<double> derivated_inputs;

    //defining this attributes in case a specific optimizer is used
	xt::xarray<double> weights_momentums;
	xt::xarray<double> weights_cache;
	xt::xarray<double> bias_momentums;
	xt::xarray<double> bias_cache;
};