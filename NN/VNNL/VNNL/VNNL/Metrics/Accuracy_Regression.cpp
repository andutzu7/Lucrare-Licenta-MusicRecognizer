#include "Accuracy_Regression.h"

Accuracy_Regression::Accuracy_Regression(xt::xarray<double> actual_values, bool reinit)
{
    if (this->precision > 0.0 || reinit)
    {
        this->precision = xt::stddev(actual_values)[0] / 250;
    }
}

xt::xarray<double> Accuracy_Regression::Compare(xt::xarray<double> predictions, xt::xarray<double> actual_values)
{
    std::vector<double> result;
    for (size_t i = 0; i < actual_values.size(); i++)
    {
        if (xt::abs(predictions - actual_values)[0] < this->precision)
        {
            result.push_back(1.0);
        }
        else
        {
            result.push_back(0.0);
        }

    }
        auto result_shape = actual_values.shape();
        return xt::adapt(result, result_shape);
}