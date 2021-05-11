#pragma once
#include <xtensor/xarray.hpp>
#include "Layer.h"
//o sa am f mult de bobinat la asta ca sa accepte np 
//arrayuri cu wav malversationated files
class InputLayer : public Layer 
{
public:
void Forward(xt::xarray<double> inputs);
//de facut private
public:
xt::xarray<double> outputs;
};