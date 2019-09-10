#include "armadillo"

template<typename T>
class Layer {
    private:
        arma::field<arma::cube> input;
        T kernel;
    public:
        arma::field<arma::cube> foward();
};