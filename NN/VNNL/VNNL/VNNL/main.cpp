#include <iostream>
#include <math.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xadapt.hpp>
#include "Layers/DenseLayer.h"
#include "Layers/Layer_Type.h"
#include "Optimizers/SGD_Optimizer.h"


using namespace std;
//fix code const correctness
/// TODO !!!! de verificat outputul la Optimizer
//de verificat sa nu fi uitat parametrii impliciti la functii prin cppuri
//to remember: for optimizers and prolly others the pipeline will be like that :
//u got a layer. the layer is then passed as a parameter for the optimizer
//the output optimized output is then assigned back to the layer (not the smoothest metthod i know)
//note for self -> since keepdims isnt availabale, to emulate it just reshape the output

int main(int argc, char *argv[])
{
	xt::xarray<double> gradient = {{0.01764052, 0.00400157, 0.00978738, 0.02240893, 0.01867558},
								   {-0.00977278, 0.00950088, -0.00151357, -0.00103219, 0.00410599},
								   {0.00144044, 0.01454273, 0.00761038, 0.00121675, 0.00443863},
								   {0.00333674, 0.01494079, -0.00205158, 0.00313068, -0.00854096},
								   {-0.0255299, 0.00653619, 0.00864436, -0.00742165, 0.02269755}};

	DenseLayer dl(5,5);
	dl.weights = gradient;
	dl.Forward(gradient);
	dl.Backward(gradient);

	cout<<dl.weights<<endl;
	//i know, its an ugly way of writing code
	SGD_Optimizer ag;
	ag.dense_layer = dl;
	ag.Update_Params(Layer_Type::dense_layer);
	dl = ag.dense_layer;	
	cout<<dl.weights;

	return 0;
}