#include <armadillo>

#include "activation.h"

using namespace arma;

namespace ops {
    void Activation::active(arma::field<arma::cube> &x, double (*f)(double)) {
        for (int i=0; i < x.n_elem; i++)
            x(i).transform([&](double val) {
                return f(val);
            });
    }
}