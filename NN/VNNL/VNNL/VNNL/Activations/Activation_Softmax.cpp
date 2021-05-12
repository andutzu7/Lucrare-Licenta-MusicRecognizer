#include "Activation_Softmax.h"
#include <iostream>
#include <xtensor/xadapt.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>

void Activation_Softmax::Forward(xt::xarray<double> inputs)
{
    this->inputs = inputs;

    auto max_keep_dims = [](xt::xarray<double> gradient)
    {
        auto shape = gradient.shape();
        std::pair<int, int> shape_pair((int)xt::adapt(shape)[0], (int)xt::adapt(shape)[1]);

        double max = -999999;
        int width = 0;
        std::vector<double> rez;
        for (int i = 0; i < gradient.size(); i++)
        {
            if (gradient[i] > max)
            {
                max = gradient[i];
            }
            width++;

            if (width == shape_pair.second)
            {
                rez.push_back(max);
                max = -999999;
                width = 0;
            }
        }
        std::vector<std::size_t> rez_shape = {rez.size(), 1};
        return (xt::adapt(rez, rez_shape));
    };

    auto sum_keep_dims = [](xt::xarray<double> gradient)
    {
        xt::xarray<double> sum_unregulated = xt::sum(gradient, 1);
        auto rez = sum_unregulated.reshape({sum_unregulated.size(), 1});
        return rez;
    };
    xt::xarray<double> exp_values = xt::exp(inputs - max_keep_dims(inputs));

    xt::xarray<double> probabilites = exp_values / sum_keep_dims(exp_values);

    this->outputs = probabilites;
}

void Activation_Softmax::Backward(xt::xarray<double> derivated_values)
{

    this->derivated_inputs = xt::empty_like(derivated_values);
    int row = 0;
    auto shape = derivated_values.shape();
    std::pair<int, int> shape_pair((int)xt::adapt(shape)[0], (int)xt::adapt(shape)[1]);
    std::vector<xt::xarray<double>> stl_result_cointainer;
    for (int i = 0; i < shape_pair.first; i++)
    {
        auto diagflat = [](xt::xarray<double> gradient)
        {
            auto shape = gradient.shape();
            std::pair<int, int> shape_pair((int)xt::adapt(shape)[0], (int)xt::adapt(shape)[1]);
            int dim = std::max(shape_pair.first, shape_pair.second);
            xt::xarray<double> aux = xt::arange(dim);
            xt::xarray<double> result = xt::diag(aux);
            for (size_t i = 0; i < dim; i++)
            {
                result(i, i) = gradient(i, 0);
            }
            return result;
        };

        std::vector<double> row;
        std::vector<double> derivated_row;
        for (int j = 0; j < shape_pair.second; j++)
        {
            row.push_back(this->outputs(i, j));
            derivated_row.push_back(derivated_values(i, j));
        }
        std::vector<std::size_t> single_output_shape = {1, row.size()};
        xt::xarray<double> single_output = xt::adapt(row, single_output_shape);

        std::vector<std::size_t> single_derivated_value_shape = {derivated_row.size()};
        xt::xarray<double> single_derivated_values = xt::adapt(derivated_row, single_derivated_value_shape);

        single_output = single_output.reshape({-1, 1});

        xt::xarray<double> jacobian_matrix = diagflat(single_output) - xt::linalg::dot(single_output, xt::transpose(single_output));

        xt::xarray<double> result = xt::linalg::dot(jacobian_matrix, single_derivated_values);

        stl_result_cointainer.push_back(result);
    }
    //very naive workaround
    std::vector<double> derivated_inputs_container;
    for (const auto& element : stl_result_cointainer)
    {
        for (size_t element_index = 0; element_index < element.size(); element_index++)
        {
            derivated_inputs_container.push_back(element[element_index]);
        }
    }
    std::vector<std::size_t> final_shape = {5,5}; 
    this->derivated_inputs = xt::adapt(derivated_inputs_container,final_shape);
}
xt::xarray<double> Activation_Softmax::Prediction(xt::xarray<double> outputs)
{
    return xt::argmax(outputs, 1);
}