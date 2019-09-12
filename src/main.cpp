#include <iostream>
#include <armadillo>

#include "nn_ops.h"
#include "activation.h"
#include "f.h"

using namespace std;
using namespace arma;
using namespace ops;
using namespace f;

int main() {    
    // arma::mat a = {
    //      {2,3.2,6},
    //      {1,5,-2},
    //      {5,1,7}
    //  };

    // arma::mat kernel = {
    //     {1,-1,2}, 
    //     {1,1,-2},
    //     {1,5,6},   
    // };

    // arma::mat c(arma::linspace(-255,0,256));
    // arma::mat d = arma::reshape(c, 16, 16).t();

    arma::mat e(arma::linspace(0,262143,262144));
    arma::mat f = arma::reshape(e, 512, 512).t();

    arma::mat kernel = {
        {1,-1,2}, 
        {1,1,-2},
        {1,0,0}
    };

    f::Conv2d::conv2d(f, kernel, Padding::SAME, 2).print();
    
    return 0;
}
