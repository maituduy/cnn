#include "add.h"

namespace layer {

    Add::Add(Layer* a, Layer* b): Layer(false) {
        this->a = a;
        this->b = b;
    }

    Add::Add(const Add& layer): Layer(layer) {
        this->a = layer.a;
        this->b = layer.b;
    }

    Layer* Add::clone() const {
        return new Add(*this);
    }

    void Add::initialize_config() {
        config["output_shape"] = config["input_shape"];
    }

    void Add::foward() {
        this->output = arma::field<arma::cube>(get_attr<Shape>("output_shape").batch);

        for (int i = 0; i < this->output.n_elem; ++i)
            this->output(i) = this->a->get_output()(i) + this->b->get_output()(i);
    }

    const char* Add::classname() {
        return "Add";
    }

}