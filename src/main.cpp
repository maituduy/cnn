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
         {2,3.2,6},
         {1,5,-2},
         {5,1,7}
     };

    mat kernel = {
        {1,-1,2}, 
        {1,1,-2},
        {1,5,6},   
    };

    Conv2d_Transpose::conv2d_transpose(a,kernel,Padding::VALID, 1).print();
    cout << Common::pool(kernel, PoolingMode::MAX) << " " << Common::pool(kernel, PoolingMode::AVERAGE) ;
    return 0;
}
