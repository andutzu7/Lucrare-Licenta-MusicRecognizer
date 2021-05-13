#include "Accuracy.h"

double Accuracy::Calculate(xt::xarray<double> predictions, xt::xarray<double> actual_values)
{

    xt::xarray<double> comparisons = Compare(predictions, actual_values);
    double accuracy = xt::mean(comparisons)();


    this->accumulation_sum += xt::sum(comparisons)[0];
    this->accumulation_count += (int)xt::adapt(comparisons.shape())[1];

    return accuracy;
}

double Accuracy::Calculate_accumulated()
{
    double accuracy = this->accumulation_sum / this->accumulation_count;
    return accuracy;
}

void Accuracy::New_Pass()
{
    this->accumulation_count = 0;
    this->accumulation_sum = 0.0;
}