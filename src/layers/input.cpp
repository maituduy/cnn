#include "armadillo"
#include "layer.h"

namespace layer {

    class Input: public Layer {
        Dict config;
        
        public:
            Input(Shape input_shape) {
                
                this->config["output_shape"] = input_shape;
                Layer::set_config(this->config);
            }

            void inject(const arma::field<arma::cube> &input) {
                Layer::set_output(input);
            }
    };
}