#include "layer.h"

namespace layer {
    class BatchNormalization: public Layer {
        float epsilon = 1e-3;
        public:
            BatchNormalization(
                Layer* pre_layer
            
            ): Layer(pre_layer, true) {
                config["output_shape"] = config["input_shape"];
                initialize();
            }

            void initialize() {
                // [gamma, beta, mean , variance]
                int c = get_attr<Shape>("input_shape").c;
                this->weights.push_back(arma::ones(c));
                this->weights.push_back(arma::zeros(c));
                this->weights.push_back(arma::zeros(c));
                this->weights.push_back(arma::ones(c));

            }

            void foward() {
                // 0.4966   0.5025   0.5023   0.5022   0.5022   0.5003   0.4961
                // 0.4920   0.4953   0.4950   0.4950   0.4950   0.4925   0.4944
                // 0.4931   0.4953   0.4951   0.4951   0.4951   0.4897   0.4868
                // 0.4931   0.4953   0.4951   0.4951   0.4951   0.4896   0.4868
                // 0.4931   0.4953   0.4950   0.4950   0.4950   0.4897   0.4869
                // 0.4942   0.4969   0.4971   0.4971   0.4971   0.4922   0.4884
                // 0.4894   0.4928   0.4926   0.4926   0.4926   0.4884   0.4856
                this->output = *this->input;
                
                auto gamma = std::get<arma::vec>(this->weights[0]);
                auto beta = std::get<arma::vec>(this->weights[1]);
                auto mean = std::get<arma::vec>(this->weights[2]);
                auto variance = std::get<arma::vec>(this->weights[3]);

                this->output.for_each([&](arma::cube& x) {
                    for (int i=0; i<x.n_slices; i++)
                        x.slice(i) = (x.slice(i) - mean[i]) / sqrt(variance(i) + epsilon) * gamma[i] + beta[i];
                });
            }

            const char* classname() { return "BatchNormalization";}
    };
}