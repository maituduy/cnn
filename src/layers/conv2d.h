#pragma once
#include "layer.h"

namespace layer {

    class Conv2d: public Layer {
        int n_filters, kernel_size;
        Padding padding;
        int stride;
        Func activation;
        public:
            Conv2d(
                int n_filters, 
                int kernel_size, 
                Padding padding = Padding::SAME,
                int stride = 1, 
                Func activation = Func::NONE
                
            );
            Conv2d(const Conv2d& layer);
            Layer *clone() const;

            void initialize_weights();
            void initialize_config();
            void foward();
            const char* classname();
                
    };
}