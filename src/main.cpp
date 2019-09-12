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
    arma::mat a = {
         {2,3.2,6},
         {1,5,-2},
         {5,1,7}
     };

    arma::mat kernel = {
        {1,-1,2}, 
        {1,1,-2},
        {1,5,6},   
    };

    arma::mat c(arma::linspace(-255,0,256));
    arma::mat d = arma::reshape(c, 16, 16).t();

    arma::mat e(arma::linspace(0,255,256));
    arma::mat f = arma::reshape(e, 16, 16).t();

    arma::cube cuin(16,16,2);
    cuin.slice(0) = d;
    cuin.slice(1) = f;

    arma::field<cube> input(1);
    input(0) = cuin;

    // Pooling2D::pool(d, 12, Padding::SAME, PoolingMode::AVERAGE_TF, 5).print();
    auto res = NnOps::pooling2d(input, 12, Padding::SAME, PoolingMode::AVERAGE_TF, 5);
    res(0).slice(0).print();
    cout << "\n";
    res(0).slice(1).print();
    
    return 0;
}
