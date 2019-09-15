#pragma once

#include "armadillo"
#include "mtype.h"

using namespace arma;
using namespace mtype;

namespace ops {
    class NnOps {
        public:
            static arma::field<cube> conv2d(const arma::field<cube> &input, arma::field<cube> kernel, Padding padding = Padding::SAME, int stride=1);
            static arma::field<cube> conv2d_transpose(const arma::field<cube> &input, arma::field<cube> kernel, Padding padding = Padding::SAME, int stride=1);
            static arma::field<cube> pooling2d(const arma::field<cube> &input, int pooling_size=2, Padding padding = Padding::SAME, PoolingMode mode = PoolingMode::MAX ,int stride=2);
            
    };
    
}