#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xexpression.hpp>
#include <xtensor/xadapt.hpp>
#include "DenseLayer.h"
//note for self -> since keepdims isnt availabale, to emulate it just reshape the output
//de fverificat Dense layer daca functioneaza p e r f e c t
using namespace std;
int main(int argc,char*argv[])
{
	DenseLayer dl(5,5);
	xt::xarray<double> gradient = xt::random::randn<double>({ 10,5 });
	dl.Forward(gradient);
	dl.Backward(gradient);
	return 0;
}