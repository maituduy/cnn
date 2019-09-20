#include "armadillo"
#include "layer.h"

namespace layer {

    class Input: public Layer {
        
        public:
            Input(Shape input_shape): Layer(nullptr, false) {
                config["input_shape"] = config["output_shape"] = input_shape;
            }

            const char* classname() { return "Input";}
    };
}