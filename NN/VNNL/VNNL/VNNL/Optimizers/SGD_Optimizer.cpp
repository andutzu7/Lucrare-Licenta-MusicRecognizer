#include "SGD_Optimizer.h"

SGD_Optimizer::SGD_Optimizer(double learning_rate  , double decay , double momentum )
{
    this->learning_rate = learning_rate;
    this->current_learning_rate = learning_rate;
    this->decay = decay;
    this->iterations = iterations;
    this->momentum = momentum;
}
void SGD_Optimizer::Update_Params(Layer_Type layer_type)
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
    xt::xarray<double> weight_updates;
    xt::xarray<double> bias_updates;
    //if the weights momentum is not initialized
    if (this->momentum > 0.0)
    {
        if (l.weights_momentums.size() == 1)
        {
            l.weights_momentums = xt::zeros_like(l.weights);
            l.bias_momentums = xt::zeros_like(l.biases);
        }
        weight_updates = this->momentum * l.weights_momentums - this->current_learning_rate * l.derivated_weights;
        l.weights_momentums = weight_updates;

        bias_updates = this->momentum * l.bias_momentums - this->current_learning_rate * l.derivated_biases;
        l.bias_momentums = bias_updates;
    }

    else
    {
        weight_updates = -this->current_learning_rate * l.derivated_weights;
        bias_updates = -this->current_learning_rate * l.derivated_biases;
    }
    l.weights += weight_updates;
    l.biases += bias_updates;
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