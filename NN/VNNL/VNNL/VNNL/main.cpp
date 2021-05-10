#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xadapt.hpp>
#include "DropoutLayer.h"
using namespace std;
//fix code const correctness
//note for self -> since keepdims isnt availabale, to emulate it just reshape the output
int main(int argc, char *argv[])
{
	/*xt::xarray<double> gradient = {{0.01764052, 0.00400157, 0.00978738, 0.02240893, 0.01867558},
								   {-0.00977278, 0.00950088, -0.00151357, -0.00103219, 0.00410599},
								   {0.00144044, 0.01454273, 0.00761038, 0.00121675, 0.00443863},
								   {0.00333674, 0.01494079, -0.00205158, 0.00313068, -0.00854096},
								   {-0.0255299, 0.00653619, 0.00864436, -0.00742165, 0.02269755}};
	*/
	return 0;
}