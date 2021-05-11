#pragma once
#include <xtensor/xarray.hpp>
#include "Optimizer.h"
#include "../Layers/Layer_Type.h"
#include "../Layers/Layer.h"


class SGD_Optimizer : public Optimizer
{
public:
    SGD_Optimizer(double learning_rate = 1.0, double decay = 0.0, double momentum = 0.0);
   
    void Update_Params(Layer_Type layer_type);
//dont forget to make privte
public:
    double momentum;
};