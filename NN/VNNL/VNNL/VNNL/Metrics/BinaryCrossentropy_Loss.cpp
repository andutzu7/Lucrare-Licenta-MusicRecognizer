#include "BinaryCrossentropy_Loss.h"

xt::xarray<double> BinaryCrossentropy::Forward(xt::xarray<double> predictions, xt::xarray<double> actual_values)
{
    xt::xarray<double> clipped_predictions = xt::clip(predictions, 1e-7, 1 - 1e-7);

    xt::xarray<double> sample_losses = -(actual_values * xt::log(clipped_predictions) + (1 - actual_values) * xt::log(1 - clipped_predictions));

    sample_losses = xt::mean(sample_losses, -1);

    return sample_losses;
}

void BinaryCrossentropy::Backward(xt::xarray<double> derivated_valus, xt::xarray<double> actual_values)
{
    int samples_number = derivated_valus.size();
    int outputs_size = (int)xt::adapt(derivated_valus.shape())[1];
    xt::xarray<double> clipped_derivated_values = xt::clip(derivated_valus, 1e-7, 1 - 1e-7);

    this->derivated_inputs = -(actual_values / clipped_derivated_values - (1-actual_values) / (1-clipped_derivated_values));

    this->derivated_inputs = this->derivated_inputs / samples_number;
}