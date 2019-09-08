#include <iostream>
#include <armadillo>
#include "../include/nn_ops.h"
#include "../include/activation.h"
#include "../include/ops_util.h"

using namespace std;
using namespace arma;
using namespace ops;
using namespace ops_util;

int main() {    
    mat A = randu<mat>(2,3);
    mat B = randu<mat>(2,3);
    // arma::field<cube> x(1);
    
    // x(0) = A;
    // x.print();
    // Activation::active(x, Activation::sigmoid);
    
    // x.print();
    A.print();
    B.print();
    mat C = ops_util::merge_cols(A,B);
    C.print();
    return 0;
}
