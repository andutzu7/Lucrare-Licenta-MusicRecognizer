#pragma once
#include <xtensor/xarray.hpp>
#include "../Layers/Layer_Type.h"
#include "../Layers/Layer.h"
// TO include more layers
#include "../Layers/DenseLayer.h"

class Optimizer
{
public:
    void Pre_Update_Params();
    void Post_Update_Params();
    //to make private
public:
    //aici trebuiesc adaugate layere
    DenseLayer dense_layer;

    double learning_rate;
    double current_learning_rate;
    double decay;
    double iterations;
};