#pragma once
#include <vector>
#include <utility>
#include <string>
#include <xtensor/xarray.hpp>
#include "Layers/Layer.h"
#include "Layers/InputLayer.h"
#include "../Metrics/Loss.h"
#include "../Optimizers/Optimizer.h"
#include "../Metrics/Accuracy.h"
//de facut destructor si curatenie dupa pointeri :(((

class Model
{

public:
    Model() = default;
    void Add(Layer *layer);
    void Set(Loss *loss = nullptr, Optimizer *optimizer = nullptr, Accuracy *accuracy=nullptr);
    void Finalize();
    void Train(xt::xarray<double> X_data, std::vector<std::string> y_labels,xt::xarray<double> validation_data ,size_t epochs = 1, size_t batch_size = 0, size_t print_every = 1);
    void Evaluate(xt::xarray<double> X_val,std::vector<std::string> y_val,size_t batch_size);
    xt::xarray<double> Predict(xt::xarray<double> X_val,size_t batch_size = 0);
    Layer* Forward(xt::xarray<double> X_data);
    void Backward(xt::xarray<double> output,std::vector<std::string> y_labels);
    std::vector<std::pair<xt::xarray<double>,xt::xarray<double>>> Get_Parameters();
    //void Set_Parameters() 
    void Save_Parameters(const std::string path);
    void Load_Parameters(const std::string path);
    void Save(const std::string path);
    static Model Load(const std::string path);
private:
    std::vector<Layer *> layers;
    std::vector<Layer *> trainable_layers;
    Layer* output_layers_activation;
    double softmax_classifier_output;
    Loss *loss;
    Optimizer *optimizer;
    Accuracy *accuracy;
    InputLayer inputLayer;
};