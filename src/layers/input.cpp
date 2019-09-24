#include "input.h"

namespace layer {

    Input::Input(Shape input_shape): Layer(false) {
        config["input_shape"] = config["output_shape"] = input_shape;
        this->pre_layer = nullptr;
    }

    Input::Input(const Input& layer): Layer(layer) {}

    Layer* Input::clone() const {
        return new Input(*this);
    }
            
    const char* Input::classname() { 
        return "Input";
    }
}