#include "Optimizer.h"

void Optimizer::Pre_Update_Params()
{
    if (this->decay > 0.0)
    {
        this->current_learning_rate = this->learning_rate * (1.0 / (1.0 + this->decay * this->iterations));
    }
}

void Optimizer::Post_Update_Params()
{
    this->iterations = this->iterations+1;
}