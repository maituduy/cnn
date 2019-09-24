#include "activation.h"

namespace layer {
    Activation::Activation(Func func): Layer(false) {
        this->func = func;
    }

    Activation::Activation(const Activation& layer): Layer(layer) {
        this->func = layer.func;
    }

    Layer* Activation::clone() const {
        return new Activation(*this);
    }

    void Activation::foward() {
        this->output = *this->input;
        f::Activation::active(&this->output, this->func);

    }

    void Activation::initialize_config() {
        this->config["output_shape"] = this->config["input_shape"];
    }

    const char* Activation::classname() {
        return "Activation";
    }

}