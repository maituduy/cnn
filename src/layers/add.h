#pragma once

#include "layer.h"
#include <initializer_list>

namespace layer {
    class Add: public Layer {
        Layer *a;
        Layer *b;
    public:
        Add(Layer* a, Layer* b);
        Add(const Add& layer);
        Layer* clone() const;
        void initialize_config();
        void foward();
        const char* classname();
    };
}