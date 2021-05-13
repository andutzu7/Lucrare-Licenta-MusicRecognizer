#include "Loss.h"

Loss::Loss()
{
    this->regularization_loss = 0.0;
}

void Loss::Regularization_Loss(Layer_Type layer_type)
{

    Layer l;
    float weight_regularizer_l1;
    float weight_regularizer_l2;
    float bias_regularizer_l1;
    float bias_regularizer_l2;
    switch (layer_type)
    {
    case Layer_Type::dense_layer:
    {
        //de facut gettere ToT
        l.inputs = this->dense_layer.inputs;
        l.weights = this->dense_layer.weights;
        l.biases = this->dense_layer.biases;
        l.derivated_weights = this->dense_layer.derivated_weights;
        l.derivated_biases = this->dense_layer.derivated_biases;
        weight_regularizer_l1 = this->dense_layer.weight_regularizer_l1;
        weight_regularizer_l2 = this->dense_layer.weight_regularizer_l2;
        bias_regularizer_l1 = this->dense_layer.bias_regularizer_l1;
        bias_regularizer_l2 = this->dense_layer.bias_regularizer_l2;
        break;
    }
    case Layer_Type::conv_2d:
    {
        //one day
        break;
    }
    }
    if (weight_regularizer_l1 > 0)
    {
        this->regularization_loss += weight_regularizer_l1 * xt::sum(xt::abs(l.weights))[0];
    }
    if (weight_regularizer_l2 > 0)
    {
        this->regularization_loss += weight_regularizer_l2 * xt::sum(l.weights * l.weights)[0];
    }
    if (bias_regularizer_l1 > 0)
    {
        this->regularization_loss += bias_regularizer_l1 * xt::sum(xt::abs(l.biases))[0];
    }
    if (bias_regularizer_l2 > 0)
    {
        this->regularization_loss += bias_regularizer_l2 * xt::sum(l.biases * l.biases)[0];
    }
}

double Loss::getRegularizationLoss()
{
    return this->regularization_loss;
}

std::pair<xt::xarray<double>, double> Loss::Calculate(xt::xarray<double> predictions, xt::xarray<double> actual_values, bool include_regularization)
{
    xt::xarray<double> sample_losses = Forward(predictions, actual_values);
    xt::xarray<double> data_loss = xt::mean(sample_losses);
    this->accumulation_sum += xt::sum(sample_losses)[0];
    this->accumulation_count += sample_losses.size();

    if (include_regularization)
    {
        return std::pair<xt::xarray<double>, double>(data_loss, getRegularizationLoss());
    }
    else
    {
        return std::pair<xt::xarray<double>, double>(data_loss, 0.0);
    }
}

std::pair<xt::xarray<double>, double> Loss::Calculate_Accumulated(xt::xarray<double> predictions, xt::xarray<double> actual_values, bool include_regularization)
{
    xt::xarray<double> data_loss = this->accumulation_sum / this->accumulation_count;

    if (include_regularization)
    {
        return std::pair<xt::xarray<double>, double>(data_loss, getRegularizationLoss());
    }
    else
    {
        return std::pair<xt::xarray<double>, double>(data_loss, 0.0);
    }
}

void Loss::New_Pass()
{
    this->accumulation_count = 0;
    this->accumulation_count = 0;
}