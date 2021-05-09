#pragma once
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include "Neuron.h"
class DenseLayer
{
public:
	DenseLayer(size_t nr_inputs , size_t nr_neurons,
		float weight_regularizer_l1 = 0, float weight_regularizer_l2 = 0,
		float bias_regularizer_l1 = 0,float bias_regularizer_l2 = 0);
	void Forward(xt::xarray<double> inputs);
	void Backward(xt::xarray<double> gradient_values);

private:
	xt::xarray<double> weights;
	xt::xarray<double> biases;
	xt::xarray<double> inputs;
	xt::xarray<double> outputs;
	xt::xarray<double> derivated_values;
	xt::xarray<double> derivated_weights;
	float weight_regularizer_l1 = 0;
	float weight_regularizer_l2 = 0;
	float bias_regularizer_l1 = 0;
	float bias_regularizer_l2 = 0;
};


