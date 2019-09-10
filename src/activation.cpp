#include "activation.h"
#include "armadillo"

using namespace arma;

namespace ops {
    void Activation::active(arma::field<arma::cube> &x, double (*f)(double)) {
        for (int i=0; i < x.n_elem; i++) {
            x(i).for_each([&](arma::cube::elem_type& val) { 
                val = (*f)(val);
            });
        }
    }
}
    