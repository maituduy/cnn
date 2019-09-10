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

    return 0;
}
