#include "Model.h"
void Model::Add(Layer *layer)
{
    this->layers.push_back(layer);
}
void Model::Set(Loss *loss, Optimizer *optimizer, Accuracy *accuracy)
{
    if (loss != nullptr)
    {
        this->loss = loss;
    }
    if (optimizer != nullptr)
    {
        this->optimizer = optimizer;
    }
    if (accuracy != nullptr)
    {
        this->accuracy = accuracy;
    }
}
void Model::Finalize()
{
    this->inputLayer = InputLayer();
    size_t layer_count = this->layers.size();
    for (size_t i = 0; i < layer_count; i++)
    {
        
    }
}
void Model::Train(xt::xarray<double> X_data, std::vector<std::string> y_labels, xt::xarray<double> validation_data, size_t epochs = 1, size_t batch_size = 0, size_t print_every = 1)
{
}
void Model::Evaluate(xt::xarray<double> X_val, std::vector<std::string> y_val, size_t batch_size)
{
}
xt::xarray<double> Predict(xt::xarray<double> X_val, size_t batch_size = 0)
{
}
Layer *Forward(xt::xarray<double> X_data)
{
}
void Backward(xt::xarray<double> output, std::vector<std::string> y_labels)
{
}
std::vector<std::pair<xt::xarray<double>, xt::xarray<double>>> Get_Parameters()
{
}
void Save_Parameters(const std::string path)
{
}
void Load_Parameters(const std::string path)
{
}
void Save(const std::string path)
{
}
static Model Load(const std::string path)
{
}