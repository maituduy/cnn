#pragma once

#include "layer.h"

namespace layer {
    class Activation: public Layer {
        Func func;
    public:
        Activation(Func func);
        Activation(const Activation& layer);
        Layer* clone() const;
        void foward();
        void initialize_config();
        const char* classname();
    };
}