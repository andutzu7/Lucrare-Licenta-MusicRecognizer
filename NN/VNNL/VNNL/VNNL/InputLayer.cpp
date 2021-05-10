#include "InputLayer.h"

void InputLayer::Forward(xt::xarray<double> inputs)
{
    this->outputs = inputs;
}