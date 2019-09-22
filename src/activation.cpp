#include "activation.h"

using namespace arma;

namespace ops {
    void Activation::active(arma::field<arma::cube> *x, mtype::Activation activation) {
        
        std::cout << activation;
        double (*f)(double);
        switch (activation) {
            case mtype::Activation::RELU:
                f = Activation::relu;
                break;
            
            case mtype::Activation::SIGMOID:
                f = Activation::sigmoid;
                break;

            default:
                f = nullptr;
                break;
        }
        std::cout << activation;
        if (f) {
            for (int i=0; i < x->n_elem; i++)
                x->at(i).transform([&](double val) {
                    return f(val);
                });
        }
        
    }
}