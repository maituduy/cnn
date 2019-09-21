#include "layer.h"

namespace layer {
    class BatchNormalization: public Layer {
        float epsilon = 1e-3;
        public:
            BatchNormalization(): Layer(true) {}
            BatchNormalization(const BatchNormalization& layer): Layer(layer) {}
            
            BatchNormalization* clone() const {
                return new BatchNormalization(*this);
            }

            void initialize_weights() {
                // [gamma, beta, mean , variance]
                int c = get_attr<Shape>("input_shape").c;
                this->weights.push_back(arma::ones(c));
                this->weights.push_back(arma::zeros(c));
                this->weights.push_back(arma::zeros(c));
                this->weights.push_back(arma::ones(c));

            }

            void initialize_config() {
                config["output_shape"] = config["input_shape"];
            }

            void foward() {
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