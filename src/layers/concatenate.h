#pragma once

#include "layer.h"
#include <initializer_list>

namespace layer {
    class Concatenate: public Layer {
        std::vector<arma::field<arma::cube>*> list;
        std::initializer_list<Layer*> init_list;
        public:
            Concatenate(std::initializer_list<Layer*> list);
            Concatenate(const Concatenate& layer);
            Layer* clone() const;
            void initialize_config();
            void foward();
            const char* classname();
    };
}