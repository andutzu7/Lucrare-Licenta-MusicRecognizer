#include "Accuracy_Categorical.h"

Accuracy_Categorical::Accuracy_Categorical(bool binary)
{
    this->binary = binary;
}

xt::xarray<double> Accuracy_Categorical::Compare(xt::xarray<double> predictions , xt::xarray<double> actual_values)
{
    
    if (!this->binary && actual_values.shape().size() ==2)
    {
            actual_values = xt::argmax(actual_values,1);
    }
    std::vector<double> result;
    double epsilon = 0.0000005;
    for(size_t i = 0 ; i< actual_values.size(); i++)
    {
        if(fabs(predictions[i] - actual_values[i])<=epsilon*fabs(predictions[i]))
        {
            result.push_back(1.0);
        }
        else
        {
            result.push_back(0.0);
        }
    }
    
    auto result_shape = actual_values.shape();
    return xt::adapt(result,result_shape);
}