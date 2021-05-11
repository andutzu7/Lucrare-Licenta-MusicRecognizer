#pragma once
#include <xtensor/xarray.hpp>
#include "Optimizer.h"
#include "../Layers/Layer_Type.h"
#include "../Layers/Layer.h"



class Adagrad_Optimizer : public Optimizer
{
public:
    Adagrad_Optimizer(double learning_rate = 1.0, double decay = 0.0, double epsilon = 1e-7);
    void Update_Params(Layer_Type layer_type);
//dont forget to make privte
public:
    double epsilon;
};