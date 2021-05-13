#include "MeanSquaredError_Loss.h"

xt::xarray<double> MeanSquaredError::Forward(xt::xarray<double> predictions, xt::xarray<double> actual_values)
{

    xt::xarray<double> sample_losses = xt::mean(xt::square(actual_values-predictions),-1);

    return sample_losses;
}

void MeanSquaredError::Backward(xt::xarray<double> derivated_valus, xt::xarray<double> actual_values)
{
    int samples_number = derivated_valus.size();
    int outputs_size = (int)xt::adapt(derivated_valus.shape())[1];

    this->derivated_inputs = -2 * (actual_values - derivated_valus) / outputs_size;


}