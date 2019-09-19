#include "armadillo"
#include "layer.h"

namespace layer {

    class Input: public Layer {
        
        public:
            Input(Shape input_shape): Layer(nullptr, false) {

                Dict config;
                config["output_shape"] = input_shape;
                this->set_config(config);
            }

            const char* classname() { return "Input";}
    };
}