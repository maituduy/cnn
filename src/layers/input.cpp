#include "armadillo"
#include "layer.h"

namespace layer {

    class Input: public Layer {
        
        public:
            Input(Shape input_shape): Layer(false) {
                config["input_shape"] = config["output_shape"] = input_shape;
                this->pre_layer = nullptr;
            }

            const char* classname() { return "Input";}
    };
}