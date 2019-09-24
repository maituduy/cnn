#pragma once

#include "layer.h"

namespace layer {
    class BatchNormalization: public Layer {
        float epsilon = 1e-3;
        public:
            BatchNormalization();
            BatchNormalization(const BatchNormalization& layer);
            Layer* clone() const;
            void initialize_weights();
            void initialize_config();
            void foward();
            const char* classname();
    };
}