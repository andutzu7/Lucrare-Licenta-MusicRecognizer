#include <xtensor/xarray.hpp>

class Activation
{
public:
    virtual void Forward(xt::xarray<double> inputs) = 0;
    virtual void Backward(xt::xarray<double> derivated_values) = 0;
    virtual xt::xarray<double> Prediction(xt::xarray<double> outputs) = 0;
//one day make these private
public:
    xt::xarray<double> inputs;
    xt::xarray<double> outputs;
    xt::xarray<double> derivated_inputs;
};