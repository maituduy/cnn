#include <armadillo>

using namespace arma;

namespace ops {
    class NnOps {
        public:
            static arma::field<cube> conv2d(const arma::field<cube> &input, arma::field<cube> kernel, std::string shape);
            static arma::field<cube> conv2d_transpose(const arma::field<cube> &input, arma::field<cube> kernel, std::string padding, int stride[]);
    };
    
}