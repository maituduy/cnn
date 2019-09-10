#include <armadillo>

using namespace arma;

namespace ops {
    class Activation {
        public:
            static void active(arma::field<cube> &input, double (*f)(double));

            static double relu(double value) {
                return value >= 0 ? value: 0; 
            }

            static double sigmoid(double value) {
                return 1.0 / (1.0 + std::exp(value));
            }
            
    };
}

