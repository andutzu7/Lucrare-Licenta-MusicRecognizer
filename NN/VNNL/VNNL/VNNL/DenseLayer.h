#pragma once
#ifndef DENSELAYER_H
#define DENSELAYER_H
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <utility>
#include "Neuron.h"
class DenseLayer
{
public:
	DenseLayer(size_t nr_inputs , size_t nr_neurons,
	float weight_regularizer_l1 = 0.0f, float weight_regularizer_l2 = 0.0f,
	float bias_regularizer_l1 = 0.0f,float bias_regularizer_l2 = 0.0f);
	void Forward(xt::xarray<double> inputs);
	void Backward(xt::xarray<double> derivated_values);
	std::pair<xt::xarray<double>,xt::xarray<double>> getParameters();
	void setParameters(xt::xarray<double> weights,xt::xarray<double> biases);

//sa nu uit sa o fac privata
public:
	xt::xarray<double> weights;
	xt::xarray<double> biases;
	xt::xarray<double> inputs;
	xt::xarray<double> outputs;
	xt::xarray<double> derivated_biases;
	xt::xarray<double> derivated_weights;
	xt::xarray<double> derivated_inputs; 
	
	float weight_regularizer_l1 = 0.0f;
	float weight_regularizer_l2 = 0.0f;
	float bias_regularizer_l1 = 0.0f;
	float bias_regularizer_l2 = 0.0f;
};

#endif
