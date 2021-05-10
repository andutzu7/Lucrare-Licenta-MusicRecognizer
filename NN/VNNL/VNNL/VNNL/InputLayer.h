#pragma once
#include <xtensor/xarray.hpp>
//o sa am f mult de bobinat la asta ca sa accepte np 
//arrayuri cu wav malversationated files
class InputLayer
{
public:
void Forward(xt::xarray<double> inputs);
//de facut private
public:
xt::xarray<double> outputs;
};