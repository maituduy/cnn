#pragma once

#include "layer.h"

namespace layer {

    class Input: public Layer {
        
        public:
            Input(Shape input_shape);
            Input(const Input& layer);
            Layer* clone() const;
            const char* classname();
    };
}