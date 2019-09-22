#include "armadillo"
#include "layer.h"

namespace layer {

    class Input: public Layer {
        
        public:
            Input(Shape input_shape): Layer(false) {
                config["input_shape"] = config["output_shape"] = input_shape;
                std::cout << "ok";
                this->pre_layer = nullptr;
            }

            Input(const Input& layer): Layer(layer) {}

            Input* clone() const {
                return new Input(*this);
            }
            
            const char* classname() { return "Input";}
    };
}