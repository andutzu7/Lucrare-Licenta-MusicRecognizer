#include "RMSProp_Optimizer.h"

RMSProp_Optimizer::RMSProp_Optimizer(double learning_rate, double decay, double epsilon, double rho)
{
    this->learning_rate = learning_rate;
    this->current_learning_rate = learning_rate;
    this->decay = decay;
    this->iterations = 0;
    this->epsilon = epsilon;
    this->rho = rho;
}
void RMSProp_Optimizer::Update_Params(Layer_Type layer_type)
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
    xt::xarray<double> weight_cache;
    xt::xarray<double> bias_cache;
    //if the weights momentum is not initialized
    if (l.weights_cache.size() == 1)
    {
        l.weights_cache = xt::zeros_like(l.weights);
        l.bias_cache = xt::zeros_like(l.biases);
    }
    //lambda utilitary to square all elements within an array
    l.weights_cache = this->rho * l.weights_cache + (1 - this->rho) * xt::square(l.derivated_weights);
    l.bias_cache = this->rho * l.bias_cache + (1 - this->rho) * xt::square(l.derivated_biases);
    l.weights += -this->current_learning_rate * l.derivated_weights / (xt::sqrt(l.weights_cache) + this->epsilon);
    l.biases += -this->current_learning_rate * l.derivated_biases / (xt::sqrt(l.bias_cache) + this->epsilon);

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