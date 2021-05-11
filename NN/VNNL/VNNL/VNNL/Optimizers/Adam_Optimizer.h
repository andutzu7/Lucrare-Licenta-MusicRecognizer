#pragma once
#include <xtensor/xarray.hpp>
#include <math.h>
#include "Optimizer.h"
#include "../Layers/Layer_Type.h"
#include "../Layers/Layer.h"



class Adam_Optimizer : public Optimizer
{
public:
    Adam_Optimizer(double learning_rate = 0.001, double decay = 0.0, double epsilon = 1e-7,double beta_1=0.9,double beta_2=0.999);
    void Update_Params(Layer_Type layer_type);
//dont forget to make privte
public:
    double epsilon;
    double beta_1;
    double beta_2;
};