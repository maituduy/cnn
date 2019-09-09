#include <iostream>
#include <armadillo>

#include "nn_ops.h"
#include "activation.h"
#include "ops_util.h"

using namespace std;
using namespace arma;
using namespace ops;
using namespace ops_util;

int main() {    
    mat a = { 
        {3,3},
        {1,1}
    };

    mat kernel = {
        {1,2,3},
        {0,1,0},
        {2,1,2}    
    };

    ops_util::Conv2d_Transpose::conv2d_transpose(a,kernel,Padding::SAME,2).print();
    return 0;
}
