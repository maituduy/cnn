#pragma once
#include <layer.h>

namespace layer {

    class Pooling2d: public Layer {
        int kernel_size;
        PoolingMode *pooling_mode;
        Padding *padding;
        int stride;

        public:
            Pooling2d(
                int kernel_size,
                PoolingMode pooling_mode = PoolingMode::MAX,
                Padding padding = Padding::SAME,
                int stride = 1
            );

            Pooling2d(const Pooling2d& layer);
            Layer* clone() const;
            void initialize_config();
            void foward();
            const char* classname();
    };
}