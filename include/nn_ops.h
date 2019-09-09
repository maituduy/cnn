#include "armadillo"
#include "mtype.h"

using namespace arma;
using namespace mtype;

namespace ops {
    class NnOps {
        public:
            static arma::field<cube> conv2d(const arma::field<cube> &input, arma::field<cube> kernel, std::string shape);
            static arma::field<cube> conv2d_transpose(const arma::field<cube> &input, arma::field<cube> kernel, Padding padding = Padding::SAME, int stride=1);
            static arma::field<cube> max_pooling2d(const arma::field<cube> &input, Padding padding = Padding::SAME, int stride=1);
    };
    
}