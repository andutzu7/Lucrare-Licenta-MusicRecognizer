#pragma once
#include <xtensor/xarray.hpp>
#include "Optimizer.h"
#include "../Layers/Layer_Type.h"
#include "../Layers/Layer.h"



class RMSProp_Optimizer : public Optimizer
{
public:
    RMSProp_Optimizer(double learning_rate = 0.001, double decay = 0.0, double epsilon = 1e-7,double rho =0.9);
    void Update_Params(Layer_Type layer_type);
//dont forget to make privte
public:
    double epsilon;
    double rho;
};