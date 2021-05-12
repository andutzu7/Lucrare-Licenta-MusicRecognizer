#include "Adam_Optimizer.h"

Adam_Optimizer::Adam_Optimizer(double learning_rate, double decay, double epsilon, double beta_1, double beta_2)
{
    this->learning_rate = learning_rate;
    this->current_learning_rate = learning_rate;
    this->decay = decay;
    this->iterations = 0;
    this->epsilon = epsilon;
    this->beta_1 = beta_1;
    this->beta_2 = beta_2;
}
void Adam_Optimizer::Update_Params(Layer_Type layer_type)
{
    Layer l;
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
        break;
    }
    case Layer_Type::conv_2d:
    {
        //one day
        break;
    }
    }
    xt::xarray<double> weight_momentums;
    xt::xarray<double> bias_momentums;
    xt::xarray<double> weight_cache;
    xt::xarray<double> bias_cache;
    //if the weights momentum is not initialized
    if (l.weights_cache.size() == 1)
    {
        l.weights_momentums = xt::zeros_like(l.weights);
        l.bias_momentums = xt::zeros_like(l.biases);
        l.weights_cache = xt::zeros_like(l.weights);
        l.bias_cache = xt::zeros_like(l.biases);
    }
    l.weights_momentums = this->beta_1 * l.weights_momentums + (1 - this->beta_1) * l.derivated_weights;
    l.bias_momentums = this->beta_1 * l.bias_momentums + (1 - this->beta_1) * l.derivated_biases;

    xt::xarray<double> weight_momentums_corrected = l.weights_momentums / (1 - pow(this->beta_1, (this->iterations + 1)));
    xt::xarray<double> bias_momentums_corrected = l.bias_momentums / (1 - pow(this->beta_1, (this->iterations + 1)));

    l.weights_cache = this->beta_2 * l.weights_cache + (1 - this->beta_2) * xt::square(l.derivated_weights);
    l.bias_cache = this->beta_2 * l.bias_cache + (1 - this->beta_2) * xt::square(l.derivated_biases);

    xt::xarray<double> weight_cache_corrected = l.weights_cache / (1 - pow(this->beta_2, (this->iterations + 1)));
    xt::xarray<double> bias_cache_corrected = l.bias_cache / (1 - pow(this->beta_2, (this->iterations + 1)));

    l.weights += -this->current_learning_rate * weight_momentums_corrected / (xt::sqrt(weight_cache_corrected) + this->epsilon);
    l.biases += -this->current_learning_rate * bias_momentums_corrected / (xt::sqrt(bias_cache_corrected) + this->epsilon);

    switch (layer_type)
    {
    case Layer_Type::dense_layer:
    {
        //de facut gettere ToT
        this->dense_layer.inputs = l.inputs;
        this->dense_layer.weights = l.weights;
        this->dense_layer.biases = l.biases;
        break;
    }
    case Layer_Type::conv_2d:
    {
        //one day
        break;
    }
    }
}