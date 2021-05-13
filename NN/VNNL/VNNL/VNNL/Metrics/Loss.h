#pragma once
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <utility>
#include "../Layers/DenseLayer.h"
#include "../Layers/Layer_Type.h"
// adaptare py-> cpp . pentru ca nu vr sa lucreez cu memorie dinamica
//fiecare functie care itereaza in py pe iterable layers la mine va
//itera printr un layer individual
class Loss
{
public:
    Loss();
    void Regularization_Loss(Layer_Type layer_type);
    double getRegularizationLoss();
    virtual xt::xarray<double> Forward(xt::xarray<double> predictions, xt::xarray<double> actual_values)=0;
    virtual void Backward(xt::xarray<double> derivated_values, xt::xarray<double> actual_values)=0;
    std::pair<xt::xarray<double>, double> Calculate(xt::xarray<double> predictions, xt::xarray<double> actual_values,bool include_regularization = false);
    std::pair<xt::xarray<double>, double> Calculate_Accumulated(xt::xarray<double> predictions, xt::xarray<double> actual_values,bool include_regularization = false);
    void New_Pass();
    //SOS
public:
    double regularization_loss = 0.0;
    double accumulation_sum =0.0;
    int accumulation_count =0;
    xt::xarray<double> derivated_inputs;

    //aici trebuiesc adaugate layere
    DenseLayer dense_layer;
};