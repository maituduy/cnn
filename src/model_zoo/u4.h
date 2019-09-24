#pragma once

#include "batch_norm.h"
#include "concatenate.h"
#include "conv2d.h"
#include "conv2d_transpose.h"
#include "pool2d.h"
#include "input.h"
#include "activation.h"
#include "add.h"

#include "model.h"

using namespace layer;

namespace model_zoo {
    class U4 {
        private: 
            Model *model;
            Layer *batch_active(Layer *in);
            Layer *residual_block(Layer *in, int n_filters = 16, int batch_active = false);
            Layer *convolution_block(
                Layer *in,
                int n_filters, 
                int kernel_size, 
                int stride=1, 
                Padding padding = Padding::SAME,
                bool activation = true
            );

            Layer *build(Layer *input, int start_neural);
        public:
            U4(Shape input_shape, int start_neural);
            Model *get();
            
    };
}